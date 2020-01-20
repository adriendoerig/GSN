import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from skimage.transform import resize
import imgaug.augmenters as iaa
import os, imageio, random


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
    if model_name.lower() == 'conv':
        # ff conv layer
        state = tf.keras.layers.Conv2D(filters=state_n_filters, kernel_size=(3, 3), padding='same', activation=activation_fn)(x)
    elif model_name.lower() == 'convlstm':
        # convLSTM
        x = tf.keras.layers.Reshape((1, img_shape[0]//2, img_shape[1]//2, layer_1_n_filters))(x)
        state = tf.keras.layers.ConvLSTM2D(filters=state_n_filters, kernel_size=(3, 3), return_sequences=False, stateful=True, padding='same', activation=activation_fn)(x),  # return_seq=False -> returns output only at last step (= last frame))
        state = state[0]
    elif model_name.lower() == 'l_convlstm':
        # convLSTM
        x = tf.keras.layers.Conv2D(filters=layer_1_n_filters, kernel_size=(3, 3), padding='same', activation=activation_fn)(x)
        x = tf.keras.layers.Reshape((1, img_shape[0]//2, img_shape[1]//2, layer_1_n_filters))(x)
        state = tf.keras.layers.ConvLSTM2D(filters=state_n_filters*2, kernel_size=(3, 3), return_sequences=False, stateful=True, padding='same', activation=activation_fn)(x),  # return_seq=False -> returns output only at last step (= last frame))
        state = state[0]
    else:
        raise Exception('model_name {} is not understood. Please use "convlstm", "l_convlstm" or "conv"'.format(model_name))
    x = tf.keras.layers.Flatten()(state),
    x = tf.keras.layers.Dense(n_output_units, activation='softmax')(x[0])
    if use_efference_copy:
        model = tf.keras.Model(inputs=[input, is_training, efference_copy], outputs=[x, state])
    else:
        model = tf.keras.Model(inputs=[input, is_training], outputs=[x, state])
    model.compile(optimizer=optimizer, loss=loss_fn)

    # needed for saving models
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    ckpt_path = './model_checkpoints_'+RUN_ID+'/' + model_name + '_ckpt'
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)

    # Print network summary and check which layers are trainable
    model.summary()

    return model, ckpt, ckpt_path, manager


def get_saccade_model(input_state_shape, n_output_units, optimizer, model_name, RUN_ID=''):

    activation_fn = 'elu'
    output_activation_fn = 'tanh' if n_output_units == 2 else 'softmax'  # if n_output_units == 2, we are using the continuous action_coding (see saccade_generator.action_to_new_pos)
    if model_name.lower() == 'convlstm':
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_state_shape),
                                     tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=activation_fn, padding='same'),
                                     tf.keras.layers.MaxPool2D((2,2)),
                                     tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3,3), return_sequences=True, stateful=True, activation=activation_fn, padding='same'),  # return_seq=False -> returns output only at last step (= last frame))
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(n_output_units, activation=output_activation_fn)])
    elif model_name.lower() == 'cnn':
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_state_shape[-3:]),  # x,y,channels
                                     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=activation_fn, padding='same'),
                                     tf.keras.layers.MaxPool2D((2, 2)),
                                     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=activation_fn, padding='same'),
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(n_output_units, activation=output_activation_fn)])
    elif model_name.lower() == 'fully_connected':
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_state_shape[-3:]),  # x,y,channels
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(n_output_units, activation=activation_fn),
                                     tf.keras.layers.Dense(n_output_units, activation=output_activation_fn)])
    elif model_name.lower() == 'l_fully_connected':
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_state_shape[-3:]),  # x,y,channels
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(512, activation=activation_fn),
                                     tf.keras.layers.Dense(512, activation=activation_fn),
                                     tf.keras.layers.Dense(n_output_units, activation=output_activation_fn)])
    else:
        raise Exception('model_name {} is not understood. Please use "convlstm", "cnn", "fully_connected" or "l_fully_connected"'.format(model_name))

    # needed for saving models
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    ckpt_path = './model_checkpoints_'+RUN_ID+'/saccade_generator_' + model_name + '_ckpt'
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)

    # Print network summary and check which layers are trainable
    model.summary()

    return model, ckpt, ckpt_path, manager


def make_toy_dataset():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    # also create fake finetuning dataset in which the task is to classify < vs. >= 5.
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
    # Normalize the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.
    train_mean, train_std = np.mean(train_images, keepdims=False), np.std(train_images, keepdims=False)

    return train_images, train_labels, test_images, test_labels, train_mean, train_std


def normalize_img(img, train_mean, train_std):
    return (img-train_mean)/train_std.astype(np.float32)


def denormalize_img(img, train_mean, train_std):
    return (img.numpy()*train_std+train_mean).astype(np.float32)


def visualize_batch(batch_images, batch_labels, preds, loss, accuracy):
    preds = tf.argmax(preds, axis=1)
    batch_labels = tf.argmax(batch_labels, axis=1)
    fig, ax = plt.subplots(5, 5)
    for n in range(25):
        ax[n // 5][n % 5].imshow(batch_images[n, :, :, 0], cmap='gray')
        ax[n // 5][n % 5].title.set_text('label: {}, pred {}'.format(batch_labels[n], preds[n]))
    fig.suptitle('batch loss: {}, batch accuracy: {}'.format(loss, accuracy))
    plt.show()


def batch_accuracy(labels, preds):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, axis=1), tf.argmax(labels, axis=1)), tf.float32))


def make_static_sequence_batch(batch_imgs, n_frames):
    sequence_batch = []
    for img in batch_imgs:
        frames = [img]
        for t in range(n_frames - 1):
            frames.append(img)
        sequence_batch.append(tf.stack(frames))
    return tf.stack(sequence_batch)


def crop_image(img, crop_size, crop_pos, plot=False):
    outside_image = crop_pos[0]+crop_size[0]/2<0 or crop_pos[1]+crop_size[1]/2<0 or crop_pos[0]-crop_size[0]/2>=img.shape[0] or crop_pos[1]-crop_size[1]/2>=img.shape[1]
    if outside_image:  # if the crop is outside the image, return zeros
        cropped_img = np.zeros(shape=(crop_size))
    else:
        left_crop_size, right_crop_size = int(np.floor(crop_size[0]/2)), int(np.ceil(crop_size[0]/2))  # doesn't work
        left_pad, right_pad = max(crop_size), max(crop_size)
        padded_img = np.pad(img, ((left_pad, right_pad), (left_pad, right_pad), (0, 0)))
        crop_center = (crop_pos[0] + left_pad, crop_pos[1] + left_pad)
        cropped_img = padded_img[-left_crop_size+crop_center[0]:right_crop_size+crop_center[0], -left_crop_size+crop_center[1]:right_crop_size+crop_center[1], :]  # seems weird, but it is correct, because we are using the padded image, which has its axes shifted by left_pad compared to the original img
    if plot:
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(img[:,:,0])
        # draw box where the crop occured
        if outside_image:
            ax[0].set_title('Crop outside box.')
        else:
            padded_img_with_box = padded_img.copy()
            padded_img_with_box[-left_crop_size+crop_center[0]:right_crop_size+crop_center[0], -left_crop_size+crop_center[1]] = 1.
            padded_img_with_box[-left_crop_size+crop_center[0]:right_crop_size+crop_center[0], right_crop_size+crop_center[1]] = 1.
            padded_img_with_box[-left_crop_size+crop_center[0],                                -left_crop_size+crop_center[1]:right_crop_size+crop_center[1]] = 1.
            padded_img_with_box[right_crop_size+crop_center[0],                                -left_crop_size+crop_center[1]:right_crop_size+crop_center[1]] = 1.
            ax[1].imshow(padded_img_with_box[:,:,0])
        ax[2].imshow(cropped_img[:,:,0])
        fig.suptitle('Crop position: {}'.format(crop_pos))
        plt.show()
    return cropped_img


def make_random_crop_sequence_batch(batch_imgs, n_frames, crop_size, show_sequence=False):
    sequence_batch = []
    for img in batch_imgs:
        frames = [crop_image(img, crop_size, (np.random.randint(28), np.random.randint(28)))]
        for t in range(n_frames - 1):
            frames.append(crop_image(img, crop_size, (np.random.randint(28), np.random.randint(28))))
        sequence_batch.append(tf.stack(frames))
    sequence = tf.stack(sequence_batch)
    if show_sequence:
        n_sample_imgs = 5
        n_sample_timesteps = min(8, n_frames)
        fig, ax = plt.subplots(n_sample_imgs, n_sample_timesteps)
        for img in range(n_sample_imgs):
            for t in range(n_sample_timesteps):
                ax[img][t].imshow(sequence[img, t, :, :, 0])
        plt.show()
    return sequence


def save_sequence_gif(input_sequence, path, state_sequence=None, mean_across_channels=False, name=''):
    '''saves a gif showing how input sequence evolves.
    optionally, you can pass a state_sequence (a sequence of states at a network layer).
    mean_across_channels: bool. choose to show the mean across channels or each channel individually next to the input sequence
    (in the latter case, the input is shown first, then the mean, then each channel individually)'''
    os.makedirs(path, exist_ok=True)
    input_sequence = np.squeeze(input_sequence)/np.max(input_sequence)
    input_sequence[input_sequence<0] = 0
    if np.min(state_sequence)<0:  # we want positive values for the gif
        state_sequence -= np.min(state_sequence)
    if state_sequence is None:
        frames = [resize((255.0 * input_sequence[t, :, :].numpy()), (300, 300)).astype(np.uint8) for t in range(input_sequence.shape[0])]
        imageio.mimsave(path+name+'input_sequence.gif', frames, duration=0.333333)
    else:
        # input and mean state across channels side by side
        if mean_across_channels:
            state_sequence = np.mean(state_sequence, axis=-1)
            state_sequence = [state_sequence[t]/np.max(state_sequence[t]) for t in range(state_sequence.shape[0])]  # normalize between 0 and 1
            state_sequence = [resize(state_sequence[t], input_sequence.shape[1:]) for t in range(state_sequence.shape[0])]
            template = 0.25 * tf.ones((4 if i == 2 else input_sequence.shape[i] for i in range(len(input_sequence.shape))))  # just a gray rectangle
            together = tf.concat((input_sequence, template, state_sequence), axis=2)  # concatenate along the columns dimension
            frames = [resize((255.0 * together[t, :, :].numpy()), (28*4, 61*4)).astype(np.uint8) for t in range(input_sequence.shape[0])]
            imageio.mimsave(path + name + 'input_and_mean_state_sequence.gif', frames, duration=0.333333)
        else:
            im_side = input_sequence.shape[1]
            n_timesteps = input_sequence.shape[0]
            state_sequence = [state_sequence[t]/np.max(state_sequence[t]) for t in range(n_timesteps)]  # normalize between 0 and 1
            state_sequence = [resize(state_sequence[t], (im_side, im_side)) for t in range(n_timesteps)]
            state_sequence = np.array(state_sequence)
            mean_state_sequence = np.mean(state_sequence, axis=-1)
            n_channels = state_sequence.shape[-1]
            n_rows = int(np.ceil(np.sqrt(n_channels+2)))
            gap_size = 5
            out_im_side = n_rows*(im_side+gap_size)
            out_seq = np.zeros((n_timesteps, out_im_side, out_im_side))
            for t in range(n_timesteps):
                for c in range(n_channels+1):
                    if c == 0:
                        out_seq[t, :im_side, :im_side] = input_sequence[t]
                    elif c == 1:
                        out_seq[t, im_side+gap_size:2*im_side+gap_size, im_side+gap_size:2*im_side+gap_size] = mean_state_sequence[t]
                    else:
                        row_pos = int(((c*(im_side+gap_size))//out_im_side)*(im_side+gap_size))
                        col_pos = int((c*(im_side+gap_size))%out_im_side)
                        out_seq[t, row_pos:row_pos+im_side, col_pos:col_pos+im_side] = state_sequence[t, :, :, c-2]
            frames = [resize((255.0 * out_seq[t, :, :]), (out_im_side*8, out_im_side*8)).astype(np.uint8) for t in range(n_timesteps)]
            imageio.mimsave(path + name + 'input_and_full_state_sequence.gif', frames, duration=0.333333)


def plot_fix_heatmap(visual_system, saccade_generator, timesteps, state_shape, init_fix_pos, imgs, one_hot_labels, train_mean, train_std, save_path):
    labels = tf.argmax(one_hot_labels, axis=1)
    n_labels = max(labels)+1
    mean_imgs = np.zeros(((n_labels,)+imgs.shape[1:]))
    heatmaps = np.zeros(((n_labels,)+imgs.shape[1:3]))
    fig, ax = plt.subplots(2, 5)
    print('Plotting fixation heatmaps')
    for label in range(n_labels):
        print('Current label = {}'.format(label))
        these_imgs = imgs[labels==label]
        mean_imgs[label] = np.mean(these_imgs, axis=0)
        for img in these_imgs:
            visual_system.set_fix_pos(init_fix_pos())
            for t in range(timesteps):
                # take action
                old_state = visual_system.state if visual_system.state is not None else np.random.normal(0, .1, (state_shape))  # Store current state
                action = saccade_generator.get_next_action(old_state)  # Query agent for the next action
                prediction, new_state, accurate, pred_loss, center_bias_loss, new_img, new_fix_pos = visual_system.take_action(img, tf.keras.utils.to_categorical(label, n_labels), action, train_mean, train_std)  # Take action, get new state and reward (the mean and std are for normalizing)
                heatmaps[label, new_fix_pos[0], new_fix_pos[1]] += 1
        visual_system.network.reset_states()
        ax[label//(n_labels//2)][label%(n_labels//2)].imshow(30 * heatmaps[label], cmap='inferno', norm=mplt.colors.Normalize(vmin=0, vmax=1))
        ax[label//(n_labels//2)][label%(n_labels//2)].imshow(mean_imgs[label, :, :, 0], cmap='gray', norm=mplt.colors.Normalize(vmin=0, vmax=1), alpha=.5)
    plt.savefig(save_path)


def plot_fix_path(visual_system, saccade_generator, timesteps, state_shape, init_fix_pos, imgs, one_hot_labels, train_mean, train_std, save_path):
    if imgs.shape[0] > 64:
        print('Using only the 64 first images')
        imgs = imgs[:64]
    n_imgs = imgs.shape[0]

    fixations = np.zeros((imgs.shape[0], timesteps, 2))
    for i, img in enumerate(imgs):
        visual_system.set_fix_pos(init_fix_pos())
        for t in range(timesteps):
            # take action
            old_state = visual_system.state if visual_system.state is not None else np.random.normal(0, .1, (state_shape))  # Store current state
            action = saccade_generator.get_next_action(old_state)  # Query agent for the next action
            prediction, new_state, accurate, pred_loss, center_bias_loss, new_img, new_fix_pos = visual_system.take_action(img, one_hot_labels, action, train_mean, train_std)  # Take action, get new state and reward (the mean and std are for normalizing)
            fixations[i, t, :] = new_fix_pos

    sqrt_imgs = int(np.ceil(np.sqrt(n_imgs)))
    fig, ax = plt.subplots(sqrt_imgs, sqrt_imgs, figsize=(3 * sqrt_imgs, 3 * sqrt_imgs))
    for i in range(n_imgs):
        ax[i // sqrt_imgs][i % sqrt_imgs].imshow(imgs[i, :, :, 0])
        ax[i // sqrt_imgs][i % sqrt_imgs].plot(fixations[i, :, 1], fixations[i, :, 0], 'ro-', alpha=0.5)
    plt.savefig(save_path)


class VisualSystem:
    def __init__(self, network, use_efference_copy, n_distances, n_angles, action_coding, fix_pos, crop_size, loss_fn, alpha_center_bias=0):
        self.network = network  # The VisualSystem network
        self.use_efference_copy = use_efference_copy  # The visual network gets an efference copy as input if True
        self.state = None  # initialization is not important since we set the first state when taking the first action
        self.n_distances = n_distances
        self.n_angles = n_angles
        self.n_actions = n_distances * n_angles
        self.fix_pos = fix_pos  # position of first fixation
        self.crop_size = crop_size
        self.loss_fn = loss_fn
        self.alpha_center_bias = alpha_center_bias  # strength of the center bias (to avoid the network wondering off to infinity; 0 by default)
        self.action_coding = action_coding

    def action_to_new_pos(self, action):
        if self.action_coding == 'dist_angle':
            # convert the action to a distance and an angle. The actions are arranged as: [(distance0,angle0),...,(distanceN,angle0),(distance0,angle1),...,(distanceN,angleN)]
            distance, angle = tf.cast(3 * (action % self.n_distances) + 1, tf.float32), tf.cast(action // self.n_distances, tf.float32)
            self.fix_pos = (self.fix_pos[0]+int(distance*tf.math.cos(angle)), self.fix_pos[1]+int(distance*tf.math.sin(angle)))  # convert to new fixation position
        elif self.action_coding == 'xypos':
            # convert the action to an (row, col) position on the image
            fix_row, fix_col = int(action//np.sqrt(self.n_actions)), int(action%np.sqrt(self.n_actions))
            self.fix_pos = (fix_row, fix_col)  # convert to new fixation position
        elif self.action_coding == 'xypos_stride2':
            # convert the action to an (row, col) position on the image, with a stride of 2
            fix_row, fix_col = 2*int(action//np.sqrt(self.n_actions)), 2*int(action%(np.sqrt(self.n_actions)))
            self.fix_pos = (fix_row, fix_col)  # convert to new fixation position
        elif self.action_coding == 'continuous':
            # CAREFUL, THIS WILL ONLY WORK WITH A POLICY GRADIENT METHOD (I THINK). THE Q-LEARNING WILL JUST CHOOSE GREEDILY ONE OF THE TWO ACTIONS.
            # actions for x and y are two real values in [-1,1], which we convert to fix_pos. (-1,-1) = topleft corner, (1,1) = bottomright. we hijack (n_distances, n_angles) to represent the (row, col) size of the image
            fix_row, fix_col = int((action[0]+1)*self.n_distances//2), int((action[1]+1)*self.n_angles//2)  # (-1,-1) = topleft corner, (0,0) = center, (1,1) = bottomright
            self.fix_pos = (fix_row, fix_col)  # convert to new fixation position

    # take one action based on the network's current state and the current fix_pos
    # used to take actions in the "real world"
    def take_action(self, img, label, action, train_mean, train_std):

        self.action_to_new_pos(action)
        new_img = tf.expand_dims(crop_image(img, self.crop_size, self.fix_pos), 0)  # crop image at fixation, and add a batch dimension
        new_img = normalize_img(new_img, train_mean, train_std)

        if self.use_efference_copy:
            prediction, new_state = self.network(inputs=[new_img, True, tf.expand_dims(self.fix_pos, axis=0)])  # predict class based on recurrent state. This state integrates info over time thanks to stateful=True in the model definition
        else:
            prediction, new_state = self.network(inputs=[new_img, True])  # predict class based on recurrent state. This state integrates info over time thanks to stateful=True in the model definition
        label = tf.expand_dims(label, axis=0)
        pred_loss = self.loss_fn(label, prediction)  # compute classification loss
        self.state = new_state
        accurate = batch_accuracy(label, prediction)

        center_vector = (img.shape[0]//2, img.shape[1]//2)  # coordinates of shape center
        distance_from_center = tf.norm(tf.cast(tf.subtract(self.fix_pos, center_vector), tf.float32))  # compute the distance from center of our new fixation
        center_bias_loss = self.alpha_center_bias*distance_from_center  # the loss is the classification loss + a center bias (weighted by alpha)

        return prediction, self.state, accurate, pred_loss, center_bias_loss, new_img, self.fix_pos

    def set_fix_pos(self, fix_pos):
        # Note: Does not reset the network weights.
        self.fix_pos = fix_pos


class SaccadeGenerator:
    def __init__(self, network, n_distances, n_angles, learning_rate=1e-4, discount=0.95,
                 exploration_rate=1., exploration_decay=0.9, exploration_decay_steps=5000, min_exploration_rate=0.01,
                 update_target_estimator_every=None):
        self.network = network
        self.target_network = tf.keras.models.clone_model(network)
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.vars_to_train = self.network.trainable_variables
        self.n_distances = n_distances
        self.n_angles = n_angles
        self.n_actions = n_distances*n_angles
        self.discount = discount  # How much we appreciate future reward over current
        self.exploration_rate = exploration_rate  # Initial exploration rate
        self.min_exploration_rate = min_exploration_rate  # Minimal exploration rate
        self.init_exploration_rate = exploration_rate  # needed to decay the rate (see self.update())
        self.exploration_decay = exploration_decay
        self.exploration_decay_steps = exploration_decay_steps
        self.update_target_estimator_every = update_target_estimator_every  # if None do not use fixed_q_targets. Otherwise update target network every N steps

    # Ask model to estimate Q value for specific state (inference)
    def get_Q(self, state, is_target):
        # Model input: Single state
        # Model output: Array of Q values for this single state
        if is_target and self.update_target_estimator_every is not None:
            return tf.squeeze(self.target_network(state))
        else:
            return tf.squeeze(self.network(state))

    def get_next_action(self, state):
        if random.random() > self.exploration_rate:  # Explore or exploit
            return self.greedy_action(state)
        else:
            return self.random_action()

    # Which action has bigger Q-value, estimated by our model (inference).
    def greedy_action(self, state):
        # argmax picks the higher Q-value and returns the index
        return tf.argmax(self.get_Q(state, is_target=False))

    def random_action(self):
        return np.random.randint(self.n_actions)

    def train_step(self, old_state, action, reward, new_state):
        # Ask the model for the Q values of the old state (inference)
        old_state_Q_values = self.get_Q(old_state, is_target=False).numpy()

        # Ask the model for the Q values of the new state (inference)
        new_state_Q_values = self.get_Q(new_state, is_target=True).numpy()

        # Real Q value for the action we took. This is what we will train towards.
        if len(reward.shape) > 0:
            # we are using memory replay batches
            for b in range(reward.shape[0]):
                old_state_Q_values[b, action] = reward[b] + self.discount * np.amax(new_state_Q_values[b])
        else:
            old_state_Q_values[action] = reward + self.discount * np.amax(new_state_Q_values)

        # Train
        training_input = old_state
        target_output = old_state_Q_values
        training_input = (training_input-np.mean(training_input))/np.std(training_input)
        # fig, ax = plt.subplots(4,4)
        # [ax[i//4][i%4].imshow(training_input[0,:,:,i]) for i in range(16)]
        # plt.show()
        with tf.GradientTape() as saccade_generator_tape:
            network_output = self.network(training_input)
            loss = tf.keras.losses.mean_squared_error(target_output, network_output)
            grad = saccade_generator_tape.gradient(loss, self.vars_to_train)
            self.optimizer.apply_gradients(zip(grad, self.vars_to_train))
        return loss

    def update(self, old_state, new_state, action, reward, step_counter):
        # Train our model with new data
        loss = self.train_step(old_state, action, reward, new_state)

        # Finally decay our exploration_rate toward zero
        self.exploration_rate = max(self.init_exploration_rate*self.exploration_decay**(step_counter / self.exploration_decay_steps), self.min_exploration_rate)
        return loss, self.exploration_rate

    def set_exploration_rate(self, new_rate):
        self.exploration_rate = new_rate

    def add_replay_memory(self, memory):
        self.replay_memory.append(memory)

    def set_target_network(self):
        self.target_network.set_weights(self.network.get_weights())

    def get_Q_values(self, state):
        current_Q_values = self.get_Q(state, is_target=False).numpy()
        target_Q_values = self.get_Q(state, is_target=True).numpy()
        current_Q_values = tf.reshape(current_Q_values, shape=(state.shape[0], int(np.sqrt(self.n_actions)), -1))
        target_Q_values = tf.reshape(target_Q_values, shape=(state.shape[0], int(np.sqrt(self.n_actions)), -1))
        current_Q_values = 255.*current_Q_values# /tf.reduce_max(current_Q_values)
        target_Q_values = 255.*target_Q_values# /tf.reduce_max(target_Q_values)
        return current_Q_values, target_Q_values


class ReplayMemory:
    def __init__(self, replay_memory_init_size, replay_memory_size, binary_reward):
        self.memory = []
        self.memory_init_size = replay_memory_init_size
        self.memory_size = replay_memory_size
        self.binary_reward = binary_reward

    # fill replay memory with random actions at the beginning
    def fill(self, visual_system, saccade_generator, state_shape, imgs, labels, train_mean, train_std):
        for m in range(self.memory_init_size):
            this_img = np.random.randint(imgs.shape[0])
            old_state = visual_system.state if visual_system.state is not None else np.zeros((state_shape))
            this_action = saccade_generator.random_action()
            prediction, new_state, accurate, pred_loss, center_bias_loss, new_img, new_fix_pos = visual_system.take_action(imgs[this_img], labels[this_img], this_action, train_mean, train_std)  # Take action, get new state and reward (the mean and std are for normalizing)
            if self.binary_reward:
                reward = accurate
            else:
                reward = -pred_loss - center_bias_loss
            self.memory.append((old_state, new_state, this_action, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


def train_GSNs(visual_system, saccade_generator, readout_layer, timesteps, init_fix_pos,
               train_images, train_labels, train_mean, train_std, crop_size, data_augment,
               class_optimizer, visual_system_ckpt, visual_system_ckpt_path, visual_system_manager,
               RL_optimizer, saccade_generator_ckpt, saccade_generator_ckpt_path, saccade_generator_manager,
               replay_memory_init_size, replay_memory_size,
               only_last_loss, binary_reward,
               RUN_ID, n_epochs, img_batch_size, RL_batch_size):

    # load networks if they exist
    if os.path.exists(visual_system_ckpt_path):
        print('Loading visual system')
        visual_system_ckpt.restore(visual_system_manager.latest_checkpoint)
    if os.path.exists(saccade_generator_ckpt_path):
        print('Loading saccade generator')
        saccade_generator_ckpt.restore(saccade_generator_manager.latest_checkpoint)

    if data_augment:
        augment_seq = iaa.Sequential([iaa.OneOf([iaa.AdditiveGaussianNoise(scale=0.03), iaa.AdditiveLaplaceNoise(scale=0.03), iaa.Dropout(0.03)])], random_order=True)

    # tensorboard
    log_dir = './tensorboard_logdir/' + RUN_ID
    summary_writer = tf.summary.create_file_writer(log_dir)
    tf.summary.trace_on(graph=True, profiler=False)

    n_img_batches = train_images.shape[0] // img_batch_size
    save_gif_cond = 100
    save_gif_path = './outputs/' + RUN_ID
    state_shape = visual_system.network.get_layer(readout_layer).output_shape

    use_memory_replay = True if replay_memory_init_size is not None else False
    use_fixed_q_targets = True if saccade_generator.update_target_estimator_every is not None else False
    if use_memory_replay:
        replay_memory = ReplayMemory(replay_memory_init_size, replay_memory_size, binary_reward )
        replay_memory.fill(visual_system, saccade_generator, state_shape, train_images, train_labels, train_mean, train_std)
    if use_fixed_q_targets:
        saccade_generator.set_target_network()
        saccade_generator.target_network.summary()

    for epoch in range(n_epochs):
        for n in range(n_img_batches):
            with tf.GradientTape() as visual_system_tape:
                visual_system_loss, RL_mean_reward = 0., 0.
                batch_accuracy = np.zeros((timesteps,))

                for batch_img in range(img_batch_size):
                    if n%save_gif_cond == 0 and batch_img <= 5:
                        input_seq_for_gif = []
                        state_seq_for_gif = []
                        fix_seq = []
                        states_var_seq = []
                    img, label = train_images[n*img_batch_size + batch_img], train_labels[n*img_batch_size + batch_img]
                    if data_augment: img = augment_seq(images=img)
                    # plt.figure()
                    # plt.imshow(np.squeeze(img))
                    # plt.show()
                    visual_system.set_fix_pos(init_fix_pos())
                    for t in range(timesteps):

                        # Maybe update the target network
                        if use_fixed_q_targets and saccade_generator_ckpt.step % saccade_generator.update_target_estimator_every == 0 and saccade_generator_ckpt.step > 0 and t==0:
                            if (use_memory_replay and batch_img==0) or not use_memory_replay:
                                saccade_generator.set_target_network()
                                print("\nUpdated target network.")

                        # take action
                        old_state = visual_system.state if visual_system.state is not None else np.random.normal(0, .1, (state_shape))  # Store current state
                        action = saccade_generator.get_next_action(old_state)  # Query agent for the next action
                        prediction, new_state, accurate, pred_loss, center_bias_loss, new_img, new_fix_pos = visual_system.take_action(img, label, action, train_mean, train_std)  # Take action, get new state and reward (the mean and std are for normalizing)
                        # get the adequate reward, depending on the setup (e.g. binary reward, only_last_loss, etc.)
                        if binary_reward:  # 0 if wrong, 1 if correct
                            reward = accurate
                        else:  # use visual system loss as reward. loss is minimized, but reward is maximized -> flip sign
                            reward = -pred_loss - center_bias_loss
                        if only_last_loss and t < timesteps-1:
                            reward = 0
                        if only_last_loss and t == timesteps-1:
                            visual_system_loss += pred_loss
                        if not only_last_loss:
                            visual_system_loss += pred_loss
                        batch_accuracy[t] += accurate / img_batch_size

                        if use_memory_replay:
                            # clear old memories if needed, add new memory and corresponding image
                            if len(replay_memory.memory) == replay_memory.memory_size:
                                replay_memory.memory.pop(0)
                            replay_memory.memory.append((old_state, new_state, action, reward))
                        else:
                            # update RL net based on action taken, reward received etc
                            RL_loss, exploration_rate = saccade_generator.update(old_state, new_state, action, np.array(reward), saccade_generator_ckpt.step)  # Let the agent update internals
                            RL_mean_reward += reward/(img_batch_size*timesteps)
                            if saccade_generator_ckpt.step == 0 and batch_img == 0 and t == 0:
                                saccade_generator.set_target_network()
                                print("\nUpdated target network.")
                            saccade_generator_ckpt.step.assign_add(1)

                        # collect frames to save gifs every few batches
                        if n%save_gif_cond == 0 and batch_img <= 5:
                            input_seq_for_gif.append(denormalize_img(new_img[0, :, :, :], train_mean, train_std))  #
                            state_seq_for_gif.append(new_state[0])  #
                            fix_seq.append(new_fix_pos)
                            states_var_seq.append(np.std(new_state[0]))

                    visual_system.network.reset_states()  # resets the convlstm states since we have finished processing this image

                    # save gifs every few batches
                    if n%save_gif_cond == 0 and batch_img <= 5:
                        gif_name = '/epoch_{}_batch_{}_img_{}'.format(epoch,n,batch_img)
                        print('saving sequence gif to '+save_gif_path+gif_name)
                        print('fixation sequence(t): {}'.format([fix for fix in fix_seq]))
                        print('state std sequence(t): {}'.format(states_var_seq))
                        save_sequence_gif(tf.stack(tf.cast(input_seq_for_gif, tf.float32)), path=save_gif_path, state_sequence=state_seq_for_gif, name=gif_name)

                if use_memory_replay:
                    # get a batch of samples from memory, and update RL agent
                    samples = replay_memory.sample(RL_batch_size)
                    old_states_batch, new_states_batch, action_batch, reward_batch = map(np.array, zip(*samples))
                    RL_loss, exploration_rate = saccade_generator.update(np.squeeze(old_states_batch), np.squeeze(new_states_batch), action_batch, reward_batch, saccade_generator_ckpt.step)  # Let the agent update internals
                    RL_mean_reward = np.mean(reward_batch)
                    if saccade_generator_ckpt.step == 0:
                        saccade_generator.set_target_network()
                        print("\nUpdated target network.")
                    saccade_generator_ckpt.step.assign_add(1)

                # train the visual system for one step.
                # compute loss accumulated over time
                grad = visual_system_tape.gradient(visual_system_loss, visual_system.network.trainable_variables)
                class_optimizer.apply_gradients(zip(grad, visual_system.network.trainable_variables))
                visual_system_ckpt.step.assign_add(1)

                print('Epoch {} batch: {}, visual_system_loss = {}, RL_mean_reward = {}, accuracy(t) = {}.'.format(epoch, n, visual_system_loss, RL_mean_reward, batch_accuracy))
                if n % 10 == 0:  # save to tensorbaord every 10 batches.
                    with summary_writer.as_default():
                        vis_step = tf.cast(visual_system_ckpt.step, tf.int64)
                        sac_step = tf.cast(saccade_generator_ckpt.step, tf.int64)
                        tf.summary.scalar('0_class_lr', class_optimizer.lr(visual_system_ckpt.step), step=vis_step)
                        tf.summary.scalar('1_RL_lr', RL_optimizer.lr(saccade_generator_ckpt.step), step=sac_step)
                        tf.summary.scalar('2_visual_system_loss', visual_system_loss, step=vis_step)
                        tf.summary.scalar('3_RL_loss', tf.reduce_sum(RL_loss), step=sac_step)
                        tf.summary.scalar('4_saccade_generator_reward', RL_mean_reward, step=sac_step)
                        tf.summary.scalar('5_exploration_rate', exploration_rate, step=sac_step)
                        tf.summary.scalar('6_action', action, step=sac_step)
                        if use_memory_replay:
                            # we don't collect the states batches when not using replay
                            curr_Q_values, target_Q_values = saccade_generator.get_Q_values(np.squeeze(old_states_batch))
                            tf.summary.image('0_current_Q_values', tf.expand_dims(curr_Q_values, -1), max_outputs=2, step=sac_step)
                            tf.summary.image('1_target_Q_values', tf.expand_dims(target_Q_values, -1), max_outputs=2, step=sac_step)
                            tf.summary.image('2_curr_minus_target_Q_values', tf.expand_dims(curr_Q_values, -1) - tf.expand_dims(target_Q_values, -1), max_outputs=2, step=sac_step)
                            tf.summary.image('3_old_state', np.reshape(np.squeeze(old_states_batch), (old_states_batch.shape[0], 4*state_shape[1], -1, 1)), max_outputs=2, step=vis_step)
                            tf.summary.image('4_new_state', np.reshape(np.squeeze(new_states_batch), (new_states_batch.shape[0], 4*state_shape[1], -1, 1)), max_outputs=2, step=vis_step)
                            tf.summary.histogram('0_current_Q_values_hist', curr_Q_values, step=sac_step)
                            tf.summary.histogram('1_target_Q_values_hist', target_Q_values, step=sac_step)
                            tf.summary.histogram('2_curr_minus_target_Q_values_hist', curr_Q_values-target_Q_values, step=sac_step)
                            tf.summary.histogram('3_old_state_hist', old_states_batch, step=vis_step)
                            tf.summary.histogram('4_new_state_hist', new_states_batch, step=vis_step)
                        vis_layer_0_weights = visual_system.network.get_layer('conv2d').get_weights()
                        vis_layer_readout_weights = visual_system.network.get_layer(readout_layer).get_weights()
                        sac_curr_layer_0_weights = saccade_generator.network.layers[1].get_weights()  # 0 if conv sec_net, 1 if fully_connected
                        sac_curr_layer_conv2_weights = saccade_generator.network.layers[-1].get_weights()  # -3 if conv sec_net, -1 if fully_connected
                        sac_targ_layer_0_weights = saccade_generator.target_network.layers[1].get_weights()  # 0 if conv sec_net, 1 if fully_connected
                        sac_targ_layer_conv2_weights = saccade_generator.target_network.layers[-1].get_weights()  # -3 if conv sec_net, -1 if fully_connected
                        tf.summary.histogram('5a_vis_system_weights_1ker', vis_layer_0_weights[0], step=vis_step)
                        tf.summary.histogram('5b_vis_system_weights_1bias', vis_layer_0_weights[1], step=vis_step)
                        tf.summary.histogram('5c_vis_system_weights_readout0', vis_layer_readout_weights[0], step=vis_step)
                        tf.summary.histogram('5d_vis_system_weights_readout1', vis_layer_readout_weights[1], step=vis_step)
                        tf.summary.histogram('6a_sac_curr_weights_0ker', sac_curr_layer_0_weights[0], step=sac_step)
                        tf.summary.histogram('6b_sac_curr_weights_0bias', sac_curr_layer_0_weights[1], step=sac_step)
                        tf.summary.histogram('6c_sac_curr_weights_conv2ker', sac_curr_layer_conv2_weights[0], step=sac_step)
                        tf.summary.histogram('6d_sac_curr_weights_conv2bias', sac_curr_layer_conv2_weights[1], step=sac_step)
                        tf.summary.histogram('6a_sac_targ_weights_0ker', sac_targ_layer_0_weights[0], step=sac_step)
                        tf.summary.histogram('6b_sac_targ_weights_0bias', sac_targ_layer_0_weights[1], step=sac_step)
                        tf.summary.histogram('6c_sac_targ_weights_conv2ker', sac_targ_layer_conv2_weights[0], step=sac_step)
                        tf.summary.histogram('6d_sac_targ_weights_conv2bias', sac_targ_layer_conv2_weights[1], step=sac_step)

                if n % 200 == 0 and n!=0:
                    print('SAVING MODELS TO {} and {}'.format(visual_system_ckpt_path, saccade_generator_ckpt_path))
                    visual_system_manager.save()
                    saccade_generator_manager.save()
                    print('Plotting fixation heatmaps and paths')
                    saccade_generator.set_exploration_rate(0.)
                    plot_fix_heatmap(visual_system, saccade_generator, timesteps, state_shape, init_fix_pos, train_images[:1000], train_labels[:1000], train_mean, train_std, save_gif_path+'/_fix_heatmaps_training_step_{}.png'.format(sac_step))
                    plot_fix_path(visual_system, saccade_generator, timesteps, state_shape, init_fix_pos, train_images[:25], train_labels[:25], train_mean, train_std, save_gif_path+'/_fix_paths_training_step_{}.png'.format(sac_step))
                    saccade_generator.set_exploration_rate(exploration_rate)


def test_GSNs(random_fixations, visual_system, saccade_generator, readout_layer, timesteps, init_fix_pos, test_images, test_labels, train_mean, train_std,
              visual_system_ckpt, visual_system_ckpt_path, visual_system_manager,
              only_last_loss, binary_reward,
              saccade_generator_ckpt, saccade_generator_ckpt_path, saccade_generator_manager, RUN_ID):

    if os.path.exists(visual_system_ckpt_path):
        print('Loading visual system')
        visual_system_ckpt.restore(visual_system_manager.latest_checkpoint)
    if os.path.exists(saccade_generator_ckpt_path):
        print('Loading saccade generator')
        saccade_generator_ckpt.restore(saccade_generator_manager.latest_checkpoint)

    saccade_generator.set_exploration_rate(1.) if random_fixations else saccade_generator.set_exploration_rate(0.)

    save_gif_path = './sequence_gifs/' + RUN_ID
    state_shape = visual_system.network.get_layer(readout_layer).output_shape  # shape of this layer
    n_test_stimuli = 100
    test_accuracy = np.zeros((timesteps,))

    for test_img in range(n_test_stimuli):
        input_seq_for_gif = []
        state_seq_for_gif = []
        fix_seq = []
        states_var_seq = []
        visual_system_loss, test_total_reward = 0., 0.
        visual_system.set_fix_pos(init_fix_pos())
        for t in range(timesteps):
            img, label = test_images[test_img], test_labels[test_img]  # testing with random fixations should be on the training set
            old_state = visual_system.state if visual_system.state is not None else np.zeros((state_shape))  # Store current state
            action = saccade_generator.get_next_action(old_state)  # Query agent for the next action
            prediction, new_state, accurate, pred_loss, center_bias_loss, new_img, new_fix_pos = visual_system.take_action(img, label, action, train_mean, train_std)  # Take action, get new state and reward (the mean and std are for normalizing)
            reward = -pred_loss - center_bias_loss  # loss is minimized, but reward is maximized -> flip sign
            test_total_reward += reward
            if only_last_loss and t == timesteps - 1:
                visual_system_loss = pred_loss
            else:
                visual_system_loss += pred_loss
            test_accuracy[t] += accurate / float(n_test_stimuli)

            if test_img <= 10:
                input_seq_for_gif.append(denormalize_img(new_img[0, :, :, :], train_mean, train_std))  #
                state_seq_for_gif.append(new_state[0])  #
                fix_seq.append(new_fix_pos)
                states_var_seq.append(np.std(new_state[0]))
            if test_img == n_test_stimuli - 1 and t == timesteps - 1:  # Print out progress every few iterations
                print('Testing: pred_loss = {}, accuracy(t) = {}. reward = {}'.format(visual_system_loss, test_accuracy, test_total_reward))

        visual_system.network.reset_states()  # resets the convlstm states since we have finished processing this image

        if test_img <= 10:
            gif_name = '/TESTING_random_fix_{}_img_{}'.format(random_fixations, test_img)
            print('saving sequence gif to ' + save_gif_path + gif_name)
            print('fixation sequence(t): {}'.format([fix for fix in fix_seq]))
            print('state std sequence(t): {}'.format(states_var_seq))
            save_sequence_gif(tf.stack(tf.cast(input_seq_for_gif, tf.float32)), path=save_gif_path, state_sequence=state_seq_for_gif, name=gif_name)

    if random_fixations is False:
        plot_fix_heatmap(visual_system, saccade_generator, timesteps, state_shape, init_fix_pos, test_images, test_labels, train_mean, train_std, save_gif_path+'/_fix_heatmaps_test.png')
        plot_fix_path(visual_system, saccade_generator, timesteps, state_shape, init_fix_pos, test_images[:16], test_labels[:16], train_mean, train_std, save_gif_path+'/_fix_paths_test.png')

    return test_accuracy