import tensorflow as tf
import numpy as np
from helper_functions_DQN import make_toy_dataset, make_random_crop_sequence_batch, \
    get_visual_system_model, get_saccade_model, VisualSystem, SaccadeGenerator, train_GSNs, test_GSNs

########################################################################################################################
# TO THINK ABOUT
#
# USE REGULARIZATION TERM FOR DEEP Q LOSS
# USE POLICY GRADIENT AS IN MNIH ET AL? --> USE PPO OBJECTIVE
# JUST LIKE FIXED Q TARGETS, WE COULD USE A FIXED VISUAL SYSTEM?
# SAMPLE NEW_FIX FROM A PROB DISTR AROUND INTENDED LOCATION?
# NEED TD(LAMBDA) TO CAPTURE LONGER-TERM DEPENDENCIES?
# TRY FORWARD MONTE CARLO SEARCH?
# TRY FORWARD TD(LAMBDA) SEARCH?
# WHAT ABOUT BATCH_NORM IN THE NETS?
# HOW LONG SHOULD I BOTHER TRAINING?
# WE DO NOT COMPUTE LOSS ON THE INITIAL IMAGE. IS THAT OK?? (ok because we just consider transitions from one image to the next. But behaviourally weird)
# IS IT REALLY OK TO USE MEMORY REPLAY? DO WE LOOSE INTERESTING RECURRENT DYNAMICS? (if we are markov: no)
# DOES ERROR REALLY BACKPROPAGATE FOR ALL FRAMES IN THE RECOGNITION NET? OR JUST THE LAST ONE? (both are possible now, or replay_memory)
# DO I NEED TO SUM THE RL LOSS OVER THE BATCH SIZE? (99% sure that no)
#
########################################################################################################################

# choose what to run
RUN_N = 0
do_training = 1
do_testing = 1
n_epochs = 4
im_batch_size = 64
RL_batch_size = 64
only_last_loss = True  # if true, only train on the last loss (after fixating sequence).
binary_reward = True  # if true, reward is 1 if correct, 0 otherwise
data_augment = True  # if true, use data augmentation (see train_GSNs in helper_functions.py)
use_efference_copy = False

# visual system parameters
vis_net = 'convlstm'
timesteps = 6
crop_size = (9, 9, 1)
class_lr = 1e-2
class_first_decay_steps = 500

# Dataset creation
train_images, train_labels, test_images, test_labels, train_mean, train_std = make_toy_dataset()
img_size = train_images[0].shape
n_image_classes = train_labels.shape[1]  # number of image types to classify
check_random_crop_sequence = 1
if check_random_crop_sequence: make_random_crop_sequence_batch(train_images[:5], 8, crop_size)

# RL agent parameters
sac_net = 'L_fully_connected'  # L is for "large". See get_model functions in helper_functions.
readout_layer = 'conv_lst_m2d' if vis_net is 'convlstm' else 'conv2d_1'  # layer based on which the saccade generator will generate saccades
action_coding = 'xypos_stride2'  # way of coding actions 'xypos' 'xypos_stride2', 'dist_angle' or 'continuous'-> fix_pos (-1,-1) = topleft corner, (1,1) = bottomright.
(n_distances, n_angles) = (3, 10) if action_coding is 'dist_angle' else (np.ceil(img_size[0]/2), (np.ceil(img_size[1]/2))) if action_coding is 'xypos_stride2' else ((img_size[0], img_size[1]))  # number of distances and angles the saccade generator can choose from
n_actions = n_distances*n_angles if action_coding is 'dist_angle' else (np.ceil(img_size[0]/2)*(np.ceil(img_size[1]/2))) if action_coding is 'xypos_stride2' else (img_size[0]*img_size[1]) # number of actions the saccade generator can take. The actions will be arranged as: [(distance0,angle0),...,(distanceN,angle0),(distance0,angle1),...,(distanceN,angleN)]
if action_coding is 'continuous': n_action = 2  # this case is a bit different, see saccade.generator.action_to_new_pos
init_exploration_rate = 1.
exploration_decay_steps = 200
RL_lr = 1e-5
RL_first_decay_steps = 500
init_fix_pos = lambda: (np.random.randint(12, 17), np.random.randint(12, 17))  # to brings the fixation position at a random place in the region of choice
update_target_estimator_every = 20  # if none, no fixed q target, otherwise update target network every N steps
replay_memory_init_size, replay_memory_size = None, None  # 10000, 10000  # replay_memory_init_size = None -> do not use replay memory
if only_last_loss:
    replay_memory_init_size = None  # cannot use replay if we use only the last loss
if replay_memory_init_size == None:  # rescale lr decay etc to accomodate the more frequent updates
    class_lr /= 10  # more art than science
    RL_lr /= 100  # more art than science
    RL_first_decay_steps *= RL_batch_size*timesteps
    exploration_decay_steps *= RL_batch_size*timesteps
    update_target_estimator_every *= RL_batch_size*timesteps

# names for saving, etc.
RUN_ID = 'VIS_{}_EFFCOP_{}_SAC_{}_CROPS_{}_T_{}_DECLAST_{}_ACTCODE_{}_MEM_SIZE_{}_RUN_{}'.format(vis_net, use_efference_copy, sac_net, crop_size[0], timesteps, only_last_loss, action_coding,replay_memory_size, RUN_N)  # for naming saving directories

# loss & optimizer we will use for the visual system
classification_loss = tf.keras.losses.CategoricalCrossentropy()
class_lr_decayed_fn = (tf.keras.experimental.CosineDecayRestarts(class_lr, class_first_decay_steps))
RL_lr_decayed_fn = (tf.keras.experimental.CosineDecayRestarts(RL_lr, RL_first_decay_steps))
class_optimizer = tf.keras.optimizers.Adam(learning_rate=class_lr_decayed_fn)
RL_optimizer = tf.keras.optimizers.Adam(learning_rate=RL_lr_decayed_fn)

# define the neural network models (one for the visual system, one for the saccade generator reinforcement learning)
visual_system_model, visual_system_ckpt, visual_system_ckpt_path, visual_system_manager = get_visual_system_model(crop_size, 1, n_image_classes, class_optimizer, classification_loss, vis_net, use_efference_copy)  # we take one frame at a time in stateful mode (i.e., process frames one at a time, but keeping the network state between frames)
state_shape = visual_system_model.get_layer(readout_layer).output_shape  # shape of this layer
saccade_generator_model, saccade_generator_ckpt, saccade_generator_ckpt_path, saccade_generator_manager = get_saccade_model(state_shape, n_actions, RL_optimizer, sac_net)

# define the visual system and saccade generator (they are instances of classes in helper_functions.py)
visual_system = VisualSystem(visual_system_model, use_efference_copy, n_distances, n_angles, action_coding, init_fix_pos(), crop_size, classification_loss, alpha_center_bias=0)
saccade_generator = SaccadeGenerator(saccade_generator_model, n_distances, n_angles, class_lr_decayed_fn, exploration_rate=init_exploration_rate, exploration_decay_steps=exploration_decay_steps, update_target_estimator_every=update_target_estimator_every)

# training loop
if do_training:
    train_GSNs(visual_system, saccade_generator, readout_layer, timesteps, init_fix_pos,
               train_images, train_labels, train_mean, train_std, crop_size, data_augment,
               class_optimizer, visual_system_ckpt, visual_system_ckpt_path, visual_system_manager,
               RL_optimizer, saccade_generator_ckpt, saccade_generator_ckpt_path, saccade_generator_manager,
               replay_memory_init_size, replay_memory_size,
               only_last_loss, binary_reward,
               RUN_ID, n_epochs, im_batch_size, RL_batch_size)

if do_testing:
    test_accuracies = []
    for random_fixations in [False, True]:
        test_accuracies.append(test_GSNs(random_fixations, visual_system, saccade_generator, readout_layer, timesteps, init_fix_pos, test_images[:1000], test_labels[:1000], train_mean, train_std,
                               visual_system_ckpt, visual_system_ckpt_path, visual_system_manager,
                               only_last_loss, binary_reward,
                               saccade_generator_ckpt, saccade_generator_ckpt_path, saccade_generator_manager, RUN_ID))
    print('-->Testing with learned fixations: accuracy(t) = {}'.format(test_accuracies[0]))
    print('--->Testing with random fixations: accuracy(t) = {}'.format(test_accuracies[1]))
