import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from toy_dataset_functions_PPO import crop_img, normalize_img, batch_accuracy

def get_visual_system_model(img_shape, batch_size, n_output_units, optimizer, loss_fn, model_name, use_efference_copy, RUN_ID=''):

    activation_fn = 'elu'
    is_training = tf.keras.layers.Input(shape=(1,), dtype=tf.bool)
    layer_1_n_filters = 32  # number of filters in the first conv layer
    state_n_filters = 16  # number of filters for the state we decode from

    input = tf.keras.layers.Input(batch_shape=(batch_size,) + img_shape)
    x = tf.keras.layers.Conv2D(filters=layer_1_n_filters, kernel_size=(3, 3), padding='same', activation=activation_fn)(input)
    # x = tf.keras.layers.BatchNormalization()(x, is_training)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    if use_efference_copy:
        efference_copy = tf.keras.layers.Input(batch_shape=(batch_size, 2))  # inputs the nex fixation location to the network
        processed_efference_copy = tf.keras.layers.Dense((img_shape[0]//2)*(img_shape[1]//2), activation=activation_fn)(efference_copy)
        reshaped_efference_copy = tf.keras.layers.Reshape((img_shape[0]//2, img_shape[1]//2, 1))(processed_efference_copy)
        x = tf.keras.layers.Concatenate(axis=-1)([x, reshaped_efference_copy])
        layer_1_n_filters += 1  # add the efference channel
    if model_name == 'conv' or model_name == 'random_conv':
        # ff conv layer
        state = tf.keras.layers.Conv2D(filters=state_n_filters, kernel_size=(3, 3), padding='same', activation=activation_fn)(x)
    elif model_name == 'convlstm' or model_name == 'random_convlstm':
        # convLSTM
        x = tf.keras.layers.Reshape((1, img_shape[0]//2, img_shape[1]//2, layer_1_n_filters))(x)
        state = tf.keras.layers.ConvLSTM2D(filters=state_n_filters, kernel_size=(3, 3), return_sequences=False, stateful=True, padding='same', activation=activation_fn)(x),  # return_seq=False -> returns output only at last step (= last frame))
        state = state[0]
    x = tf.keras.layers.Flatten()(state),
    x = tf.keras.layers.Dense(n_output_units, activation='softmax')(x[0])
    if use_efference_copy:
        model = tf.keras.Model(inputs=[input, is_training, efference_copy], outputs=[x, state])
    else:
        model = tf.keras.Model(inputs=[input, is_training], outputs=[x, state])
    model.compile(optimizer=optimizer, loss=loss_fn)

    # needed for saving models
    ckpt_dict = {}
    ckpt_dict['ckpt'] = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    ckpt_dict['ckpt_path'] = './model_checkpoints_'+str(RUN_ID)+'/' + model_name + '_ckpt'
    ckpt_dict['manager'] = tf.train.CheckpointManager(ckpt_dict['ckpt'] , ckpt_dict['ckpt_path'] , max_to_keep=1)

    # Print network summary and check which layers are trainable
    model.summary()

    return model, ckpt_dict


def get_saccade_actor_model(input_state_shape, optimizer, n_actions, model_name, RUN_ID=''):
    # there iares only 2 output neuron, determining the (x,y) mean of a 2D gaussian in [-1,1]x[-1,1]

    activation_fn = 'elu'
    output_activation_fn = 'tanh' if n_actions == 2 else 'softmax'  # use tanh only if using continuous action coding
    if 'convlstm' in model_name.lower() :
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_state_shape),
                                     tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=activation_fn, padding='same'),
                                     tf.keras.layers.MaxPool2D((2,2)),
                                     tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3,3), return_sequences=True, stateful=True, activation=activation_fn, padding='same'),  # return_seq=False -> returns output only at last step (= last frame))
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(n_actions, activation=output_activation_fn)])
    elif 'cnn' in model_name.lower():
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_state_shape[-3:]),  # x,y,channels -> the point is that we don't take the "time" dimension of convLSTM
                                     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=activation_fn, padding='same'),
                                     tf.keras.layers.MaxPool2D((2, 2)),
                                     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=activation_fn, padding='same'),
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(n_actions, activation=output_activation_fn)])
    elif 'fully_connected' in model_name.lower():
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_state_shape[-3:]),  # x,y,channels -> the point is that we don't take the "time" dimension of convLSTM
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(512, activation=activation_fn),
                                     tf.keras.layers.Dense(n_actions, activation=output_activation_fn)])
    else:
        raise Exception('model_name {} is not understood. Please use "convlstm", "cnn" or "dense"')

    # needed for saving models
    ckpt_dict = {}
    ckpt_dict['ckpt'] = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    ckpt_dict['ckpt_path'] = './model_checkpoints_'+str(RUN_ID)+'/saccade_actor_' + model_name + '_ckpt'
    ckpt_dict['manager'] = tf.train.CheckpointManager(ckpt_dict['ckpt'] , ckpt_dict['ckpt_path'] , max_to_keep=1)

    # compile keras model & print model summary
    model.summary()

    return model, ckpt_dict


def get_saccade_critic_model(input_state_shape, optimizer, model_name, RUN_ID=''):
    # there is a single output neuron, representing the q_value

    activation_fn = 'elu'
    output_activation_fn = None
    if 'convlstm' in model_name.lower():
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_state_shape),
                                     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=activation_fn, padding='same'),
                                     tf.keras.layers.MaxPool2D((2, 2)),
                                     tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), return_sequences=True, stateful=True, activation=activation_fn, padding='same'), # return_seq=False -> returns output only at last step (= last frame))
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(1, activation=output_activation_fn)])
    elif 'cnn' in model_name.lower():
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_state_shape[-3:]),  # x,y,channels
                                     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=activation_fn, padding='same'),
                                     tf.keras.layers.MaxPool2D((2, 2)),
                                     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=activation_fn, padding='same'),
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(1, activation=output_activation_fn)])
    elif 'fully_connected' in model_name.lower():
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_state_shape[-3:]),  # x,y,channels
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(512, activation=activation_fn),
                                     tf.keras.layers.Dense(1, activation=output_activation_fn)])
    else:
        raise Exception('model_name {} is not understood. Please use "convlstm", "cnn" or "dense"')

    # needed for saving models
    ckpt_dict = {}
    ckpt_dict['ckpt'] = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    ckpt_dict['ckpt_path'] = './model_checkpoints_' + str(RUN_ID) + '/saccade_critic_' + model_name + '_ckpt'
    ckpt_dict['manager'] = tf.train.CheckpointManager(ckpt_dict['ckpt'], ckpt_dict['ckpt_path'], max_to_keep=1)

    # print network summary
    model.summary()

    return model, ckpt_dict


class VisualSystem:
    def __init__(self, network, use_efference_copy, optimizer, fix_pos, crop_size, im_size, loss_fn, n_actions, action_coding, alpha_center_bias=0):
        self.network = network  # The VisualSystem network
        self.use_efference_copy = use_efference_copy  # The visual network gets an efference copy as input if True
        self.optimizer = optimizer
        self.state = None  # initialization is not important since we set the first state when taking the first action
        self.fix_pos = fix_pos  # position of first fixation
        self.im_size = im_size
        self.crop_size = crop_size
        self.loss_fn = loss_fn
        self.n_actions = n_actions
        self.action_coding = action_coding
        self.alpha_center_bias = alpha_center_bias  # strength of the center bias (to avoid the network wondering off to infinity; 0 by default)

    def action_to_new_pos(self, action):
        if self.action_coding == 'xypos_stride2':
            # convert the action to an (row, col) position on the image, with a stride of 2
            fix_row, fix_col = 2*int(action//np.sqrt(self.n_actions)), 2*int(action%(np.sqrt(self.n_actions)))
            self.fix_pos = (fix_row, fix_col)  # convert to new fixation position
        elif self.action_coding == 'continuous':
            # actions for x and y are two real values in [-1,1], which we convert to fix_pos. (-1,-1) = topleft corner, (1,1) = bottomright.
            action = tf.squeeze(action)
            fix_row, fix_col = int((action[0]+1)*self.im_size[0]//2), int((action[1]+1)*self.im_size[1]//2)
            fix_row = fix_row if fix_row < 28 else 27
            fix_col = fix_col if fix_col < 28 else 27
            self.fix_pos = (fix_row, fix_col)  # set new fixation position

    # take one action based on the network's current state and the current fix_pos
    def take_action(self, img, label, action, train_mean, train_std):

        self.action_to_new_pos(action)
        new_img = tf.expand_dims(crop_img(img, self.crop_size, self.fix_pos), 0)  # crop image at fixation, and add a batch dimension
        new_img = normalize_img(new_img, train_mean, train_std)

        if self.use_efference_copy:
            prediction, new_state = self.network(inputs=[new_img, True, tf.expand_dims(self.fix_pos, axis=0)])  # predict class based on recurrent state. This state integrates info over time thanks to stateful=True in the model definition
        else:
            prediction, new_state = self.network(inputs=[new_img, True])  # predict class based on recurrent state. This state integrates info over time thanks to stateful=True in the model definition

        label = tf.expand_dims(label, axis=0)
        pred_loss = self.loss_fn(label, prediction)  # compute classification loss
        accurate = batch_accuracy(label, prediction)
        self.state = new_state

        center_vector = (img.shape[0]//2, img.shape[1]//2)  # coordinates of shape center
        distance_from_center = tf.norm(tf.cast(tf.subtract(self.fix_pos, center_vector), tf.float32))  # compute the distance from center of our new fixation
        center_bias_loss = self.alpha_center_bias*distance_from_center  # the loss is the classification loss + a center bias (weighted by alpha)

        return prediction, self.state, accurate, pred_loss, center_bias_loss, new_img, self.fix_pos

    def set_fix_pos(self, fix_pos):
        self.fix_pos = fix_pos


class SaccadeGenerator:
    def __init__(self, actor_network, critic_network, optimizer, n_actions, sigma):
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.optimizer = optimizer
        self.n_actions = n_actions  # NOT used with action_coding = 'continuous'
        self.sigma = tf.Variable(sigma)  # ONLY used with action_coding = 'continuous'. the std of the normal distribution from which fixations are drawn

    def get_value(self, state):
        # Input: Single state
        # Output: critic's estimation of Q-value for this state
        return tf.squeeze(self.critic_network(state))

    def get_action_log_prob_and_entropy(self, state):
        # Input: Single state
        # Output: actor's choice for the next action, and associated log probability
        if self.n_actions == 2:
            # in this case we are using continuous action coding
            mu = tf.squeeze(self.actor_network(state))  # mean of a 2d gaussian
            prob_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=[self.sigma, self.sigma])  # 2d gaussian of mean mu and std sigma
            action = tf.squeeze(prob_dist.sample(1))  # action is sampled from this 2d gaussian
            log_probs = tf.squeeze(prob_dist.log_prob(action))
            entropy = tf.squeeze(prob_dist.entropy())
        else:
            action_probs = self.actor_network(state)
            action = tf.random.categorical(action_probs, 1)
            log_probs = tf.math.log(action_probs + 1e-10)
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=-1)
        return action, log_probs, entropy
