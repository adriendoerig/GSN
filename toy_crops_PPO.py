import tensorflow as tf
import numpy as np
from models_PPO import get_visual_system_model, get_saccade_actor_model, get_saccade_critic_model, VisualSystem, SaccadeGenerator
from training_functions_PPO import train_GSNs, test_GSNs
from toy_dataset_functions_PPO import make_toy_dataset, make_random_crop_sequence_batch

########################################################################################################################
# TO THINK ABOUT
#
# CHECK ALL NORMALIZATIONS
# IF UNSTABLE, USE LARGER BATCH SIZE FOR PPO
# USE A LOWER GAMMA BECAUSE WE HAVE ONLY 6 TIMESTEPS?
# PLOT KL DIVERGENCE BETWEEN OLD AND NEW POLICY, PLOT CRITIC'S EXPLAINED VARIANCE
# REMOVE LR DECAY?
# UPDATE VIS_SYSTEM MORE OFTEN THAN SAC_GENERATOR, TO FIX MOVING TARGET PROBLEM?
# FEWER HIDDEN LAYER OR N_HIDDEN_UNITS?
#
########################################################################################################################

# choose what to run
RUN_N = 2
do_training = 1
do_testing = 1

# general parameters
timesteps = 6  # number of recurrent timesteps
n_img_epochs = 40  # epochs going through MNIST training set
img_batch_size = 16  # visual system trained every img_batch_size images, i.e., on img_batch_size*timesteps samples
n_ppo_epochs = 10  # epochs going through ppo_buffer at each ppo update
ppo_buffer_size = img_batch_size*25  # PPO gradient descent occurs every ppo_batch_size imgs, i.e., on ppo_batch_size*timesteps samples
ppo_batch_size = img_batch_size  # when we do gradient descent to update PPO policy, use this batch size
only_last_loss = True  # if true, only train on the last loss (after fixating sequence).
binary_reward = True  # if true, reward is 1 if correct, 0 otherwise NOT TESTED
data_augment = True  # if true, use data augmentation (see train_GSNs in helper_functions.py)
use_efference_copy = True  # if true, the position of next fixation is given to the visual network

# Dataset creation
train_images, train_labels, test_images, test_labels, train_mean, train_std = make_toy_dataset()
img_size = train_images[0].shape
n_image_classes = train_labels.shape[1]  # number of image types to classify

# visual system parameters
vis_net = 'convlstm'
crop_size = (9, 9, 1)
class_lr = 1e-3
class_first_decay_steps = 1000

# RL agent parameters
sac_net = 'fully_connected'
readout_layer = 'conv_lst_m2d' if vis_net is 'convlstm' else 'conv2d_1'  # layer based on which the saccade generator will generate saccades
action_coding = 'xypos_stride2'  # 'continuous' or 'xypos_stride2' (continuous not working atm)
n_actions = int((np.ceil(img_size[0]/2)*(np.ceil(img_size[1]/2)))) if action_coding == 'xypos_stride2' else 2  # (x,y) pos if continuous
RL_lr = 1e-4
RL_first_decay_steps = 1000*(ppo_buffer_size/ppo_batch_size)
gamma = 0.9  # discount factor (for ppo loss)
lmbda = 0.95  # lambda in ppo loss
sigma = .01  # std of the normal function from which fixations are drawn (see SaccadeGenerator in models_PPO.py). ONLY FOR CONTINUOUS ACTION CODING
clipping_val = 0.2  # for ppo loss
alpha_critic_loss = 0.5  # weight of critic in total ppo loss
alpha_entropy_loss = 1000  # weight of entropy in total ppo loss
init_fix_pos = lambda: (np.random.randint(12, 17), np.random.randint(12, 17))  # brings the fixation position at a random place in the region of choice

# names for saving, etc.
RUN_ID = 'VIS_{}_EFFCOP_{}_SAC_{}_CROPS_{}_T_{}_DECLAST_{}_RUN_{}'.format(vis_net, use_efference_copy, sac_net, crop_size[0], timesteps, only_last_loss, RUN_N)  # for naming saving directories

# loss & optimizer we will use for the visual system
classification_loss = tf.keras.losses.CategoricalCrossentropy()
# class_lr_decayed_fn = (tf.keras.experimental.CosineDecayRestarts(class_lr, class_first_decay_steps))
class_lr_decayed_fn = tf.keras.optimizers.schedules.ExponentialDecay(class_lr, decay_steps=class_first_decay_steps, decay_rate=0.96)
# RL_lr_decayed_fn = (tf.keras.experimental.CosineDecayRestarts(RL_lr, RL_first_decay_steps))
RL_lr_decayed_fn = tf.keras.optimizers.schedules.ExponentialDecay(class_lr, decay_steps=RL_first_decay_steps, decay_rate=0.96)
class_optimizer = tf.keras.optimizers.Adam(learning_rate=class_lr_decayed_fn)
RL_optimizer = tf.keras.optimizers.Adam(learning_rate=RL_lr_decayed_fn)

# define the neural network models (one for the visual system, one for the saccade generator reinforcement learning)
visual_system_model, visual_system_ckpt_dict = get_visual_system_model(crop_size, 1, n_image_classes, class_optimizer, classification_loss, vis_net, use_efference_copy, RUN_N)  # we take one frame at a time in stateful mode (i.e., process frames one at a time, but keeping the network state between frames)
state_shape = visual_system_model.get_layer(readout_layer).output_shape  # shape of this layer
saccade_actor, saccade_actor_ckpt_dict = get_saccade_actor_model(state_shape, RL_optimizer, n_actions, sac_net+'_actor', RUN_N)
saccade_critic, saccade_critic_ckpt_dict = get_saccade_critic_model(state_shape, RL_optimizer, sac_net+'_critic', RUN_N)

# define the visual system and saccade generator (they are instances of classes in helper_functions.py)
visual_system = VisualSystem(visual_system_model, use_efference_copy, class_optimizer, init_fix_pos(), crop_size, img_size, classification_loss, n_actions, action_coding, alpha_center_bias=0)
saccade_generator = SaccadeGenerator(saccade_actor, saccade_critic, RL_optimizer, n_actions, sigma)

# training loop
if do_training:
    train_GSNs(visual_system, saccade_generator, readout_layer, timesteps, init_fix_pos,
               train_images, train_labels, train_mean, train_std, data_augment,
               n_img_epochs, img_batch_size, n_ppo_epochs, ppo_buffer_size, ppo_batch_size, only_last_loss, binary_reward,
               gamma, lmbda, clipping_val, alpha_critic_loss, alpha_entropy_loss,
               visual_system_ckpt_dict, saccade_actor_ckpt_dict, saccade_critic_ckpt_dict, RUN_ID)

if do_testing:
    test_accuracies = []
    for random_fixations in [False, True]:
        if random_fixations:
            # define randomly connected (i.e., untrained) neural network models (one for the visual system, one for the saccade generator reinforcement learning)
            random_visual_system_model, _ = get_visual_system_model(crop_size, 1, n_image_classes, class_optimizer, classification_loss, 'random_'+vis_net, use_efference_copy)  # we take one frame at a time in stateful mode (i.e., process frames one at a time, but keeping the network state between frames)
            random_saccade_actor, _ = get_saccade_actor_model(state_shape, RL_optimizer, n_actions, 'random_'+sac_net+'_actor')
            saccade_critic, saccade_critic_ckpt_dict = get_saccade_critic_model(state_shape, RL_optimizer, 'random_'+sac_net+'_critic')

            # define the visual system and saccade generator (they are instances of classes in helper_functions.py)
            random_visual_system = VisualSystem(visual_system_model, use_efference_copy, init_fix_pos(), crop_size, img_size, classification_loss, n_actions, action_coding, alpha_center_bias=0)
            random_saccade_generator = SaccadeGenerator(saccade_actor, saccade_critic, n_actions, sigma)

            test_accuracies.append(test_GSNs(random_fixations, visual_system, saccade_generator, readout_layer, timesteps, init_fix_pos,
              test_images, test_labels, train_mean, train_std,
              visual_system_ckpt_dict, saccade_actor_ckpt_dict, saccade_critic_ckpt_dict,
              only_last_loss, binary_reward, RUN_ID))
        else:
            test_accuracies.append(test_GSNs(random_fixations, visual_system, saccade_generator, readout_layer, timesteps, init_fix_pos,
              test_images, test_labels, train_mean, train_std,
              visual_system_ckpt_dict, saccade_actor_ckpt_dict, saccade_critic_ckpt_dict,
              only_last_loss, binary_reward, RUN_ID))

    print('-->Testing with learned fixations: accuracy(t) = {}'.format(test_accuracies[0]))
    print('--->Testing with random fixations: accuracy(t) = {}'.format(test_accuracies[1]))
