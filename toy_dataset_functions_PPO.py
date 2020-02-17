import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mplt
from skimage.transform import resize
import os, imageio

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


def crop_img(img, crop_size, crop_pos, plot=False):
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
        frames = [crop_img(img, crop_size, (np.random.randint(28), np.random.randint(28)))]
        for t in range(n_frames - 1):
            frames.append(crop_img(img, crop_size, (np.random.randint(28), np.random.randint(28))))
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
                action, _, _, _ = saccade_generator.get_action_log_prob_and_entropy(old_state)  # Query agent for the next action
                prediction, new_state, accurate, pred_loss, center_bias_loss, new_img, new_fix_pos = visual_system.take_action(img, tf.keras.utils.to_categorical(label, n_labels), action, train_mean, train_std)  # Take action, get new state and reward (the mean and std are for normalizing)
                heatmaps[label, new_fix_pos[0], new_fix_pos[1]] += 1
        visual_system.network.reset_states()
        ax[label//(n_labels//2)][label%(n_labels//2)].imshow(30 * heatmaps[label], cmap='inferno', norm=mplt.colors.Normalize(vmin=0, vmax=1))
        ax[label//(n_labels//2)][label%(n_labels//2)].imshow(mean_imgs[label, :, :, 0], cmap='gray', norm=mplt.colors.Normalize(vmin=0, vmax=1), alpha=.5)
    plt.savefig(save_path)
    plt.close()


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
            action, _, _, _ = saccade_generator.get_action_log_prob_and_entropy(old_state)  # Query agent for the next action
            prediction, new_state, accurate, pred_loss, center_bias_loss, new_img, new_fix_pos = visual_system.take_action(img, one_hot_labels, action, train_mean, train_std)  # Take action, get new state and reward (the mean and std are for normalizing)
            fixations[i, t, :] = new_fix_pos

    sqrt_imgs = int(np.ceil(np.sqrt(n_imgs)))
    fig, ax = plt.subplots(sqrt_imgs, sqrt_imgs, figsize=(3 * sqrt_imgs, 3 * sqrt_imgs))
    for i in range(n_imgs):
        ax[i // sqrt_imgs][i % sqrt_imgs].imshow(imgs[i, :, :, 0])
        ax[i // sqrt_imgs][i % sqrt_imgs].plot(fixations[i, :, 1], fixations[i, :, 0], 'ro-', alpha=0.5)
    plt.savefig(save_path)
    plt.close()
