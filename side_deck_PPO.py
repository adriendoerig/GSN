from retina.common import get_rf_value, delta_lookup, fit_func
from retina.retina import warp_image
import os, matplotlib


def train_on_toy_dataset(model, n_frames, crop_type, train_data, train_labels, test_data, test_labels, batch_size, n_epochs, loss_fn, optimizer, checkpoint, checkpoint_path, saving_manager):

    checkpoint_path = checkpoint_path
    if os.path.exists(checkpoint_path):
        checkpoint.restore(saving_manager.latest_checkpoint)
        print('Checkpoint {} found, skipping training...'.format(checkpoint_path))
        return
    else:
        n_samples = train_data.shape[0]
        n_batches = n_samples//batch_size
        losses = np.zeros(n_batches*n_epochs)
        accuracies = np.zeros(n_batches*n_epochs)
        vars_to_train = model.trainable_variables
        counter = 0

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                batch_imgs = train_data[batch*batch_size:(batch+1)*batch_size]
                if crop_type == None:
                    batch_imgs = make_static_sequence_batch(batch_imgs, n_frames)
                elif isinstance(crop_type, tuple):
                    batch_imgs = make_random_crop_sequence_batch(batch_imgs, n_frames, crop_type)
                else:
                    print('please enter None (static sequence), a tuple (random crop sequence) or "LEARN" (learn cropping by reinforcement learning) for crop_type')
                if batch == 0:  # save one gif of the stimulus sequence before each epoch
                    save_sequence_gif(batch_imgs[0], path=checkpoint_path[:-4], name='before_epoch_{}'.format(epoch))
                batch_labels = train_labels[batch*batch_size:(batch+1)*batch_size]
                losses[counter], accuracies[counter] = train_step(model, batch_imgs, batch_labels, loss_fn, optimizer, vars_to_train)
                if batch % 25 == 0:
                    print('\rEpoch {}, batch {} -- loss = {}, accuracy = {}'.format(epoch, batch, losses[counter], accuracies[counter]), end=' ')
                counter += 1
                checkpoint.step.assign_add(1)
                if int(checkpoint.step) % 250 == 0 or batch == n_batches-1:
                    save_path = saving_manager.save()
                    print("\nSaved checkpoints for step {}: {} (epoch {}).".format(int(checkpoint.step), save_path, epoch))

        if test_data is not None:
            print('\nComputing performance on test set...')
            n_test_samples = test_data.shape[0]
            n_test_batches = n_test_samples//batch_size
            test_loss = 0
            test_accuracy = 0
            for batch in range(n_test_batches):
                batch_imgs = test_data[batch * batch_size:(batch + 1) * batch_size]
                if crop_type == None:
                    batch_imgs = make_static_sequence_batch(batch_imgs, n_frames)
                elif isinstance(crop_type, tuple):
                    batch_imgs = make_random_crop_sequence_batch(batch_imgs, n_frames, crop_type)
                else:
                    print('please enter None (static sequence), a tuple (random crop sequence) or "LEARN" (learn cropping by reinforcement learning) for crop_type')
                if batch == 0:
                    save_sequence_gif(batch_imgs[0], path=checkpoint_path[:-4], name='testing')
                batch_labels = test_labels[batch * batch_size:(batch + 1) * batch_size]
                test_loss += loss_fn(batch_labels, model(batch_imgs))/n_test_batches
                test_accuracy += batch_accuracy(batch_labels, model(batch_imgs))/n_test_batches
                if batch % 25 == 0:
                    print('\rTesting progress: {} %'.format(batch/n_test_batches*100), end=' ')
            print('\nTesting loss = {}\nTesting accuracy = {}'.format(test_loss, test_accuracy))

        fig, a = plt.subplots(1, 2)
        a[0].plot(range(n_batches*n_epochs), losses)
        a[1].plot(range(n_batches*n_epochs), accuracies)
        plt.show()

        return losses, accuracies


def train_step(model, batch_imgs, batch_labels, loss_fn, optimizer, vars_to_train, visualize_this_batch=False):
    tf.debugging.assert_rank(tf.squeeze(batch_labels), 2, message='please provide one_hot encoded labels')
    n_frames = batch_labels.shape[1]
    with tf.GradientTape() as tape:
        preds = model(batch_imgs)
        accuracy = batch_accuracy(batch_labels, preds)
        loss = loss_fn(batch_labels, preds)  # careful: if we have one output per timestep, this will need to change
        grad = tape.gradient(loss, vars_to_train)
        optimizer.apply_gradients(zip(grad, vars_to_train))
        if visualize_this_batch: visualize_batch(batch_imgs, batch_labels, preds, loss, accuracy)
    return loss, accuracy


def get_retina_pars(in_size, out_size):

    optimal_rf = get_rf_value(in_size, out_size)
    print('Optimal RF for input size [{0}x{0}] and output size [{1}x{1}]: {2:.2f}'.format(in_size, out_size, optimal_rf))
    rprime, r = delta_lookup(in_size, out_size, max_ratio=10.)

    # find_retina_mapping(fit_mode='quad')
    func = lambda x, a, b: a * x ** 2 + b * x
    retina_func = func
    popt, pcov = fit_func(func, rprime, r)
    retina_pars = popt

    # simulated version
    r_s = np.arange(out_size / 2 + 1, step=16)
    r_simulated = np.tile(r_s, (20, 1)).T.flatten()
    theta = np.tile(np.linspace(-np.pi, np.pi, 20), (1, len(r_s)))
    r_simulated = retina_func(r_simulated, *retina_pars)
    x_simulated = in_size / 2. + r_simulated * np.cos(theta)
    y_simulated = in_size / 2. + r_simulated * np.sin(theta)

    # real sampling version
    # xy = warp_func(xy_out, in_size, retina_func, retina_pars, shift=None)
    xy_out = np.indices((out_size, out_size))[:, ::16, ::16][:, 1:, 1:].reshape(2, -1)
    xy_out = xy_out - out_size / 2.
    r_out = np.linalg.norm(xy_out, axis=0)
    theta = np.arctan2(xy_out[1], xy_out[0])

    r = retina_func(r_out, *retina_pars)
    x = in_size / 2. + r * np.cos(theta)
    y = in_size / 2. + r * np.sin(theta)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].add_patch(matplotlib.patches.Rectangle((0, 0), in_size, in_size, fill=False))
    ax[0].scatter(x_simulated, y_simulated, color='r')
    ax[0].axvline(in_size / 2., ls='--', c='b')
    ax[0].axhline(in_size / 2., ls='--', c='b')
    ax[0].axis('equal')
    ax[0].set_title('simulated cones \n(for visualization)')

    ax[1].add_patch(matplotlib.patches.Rectangle((0, 0), in_size, in_size, fill=False))
    ax[1].scatter(x, y, color='r')
    ax[1].axvline(in_size / 2., ls='--', c='b')
    ax[1].axhline(in_size / 2., ls='--', c='b')
    ax[1].axis('equal')
    ax[1].set_title('simulated sampling')

    plt.show()

    return retina_pars


def plot_retinal_image(img_orig, in_size, out_size):
    resize_scale = 0.71
    img = resize(img_orig, np.array(resize_scale * np.array(img_orig.shape[:2]), dtype=int))
    ret_img = warp_image(img, output_size=out_size, input_size=in_size)
    fig, ax = plt.subplots(ncols=2, figsize=(10, 10))
    ax[0].imshow(img_orig)
    ax[0].set_title('Original image')
    ax[1].imshow(ret_img)
    ax[1].set_title('Retinal image')
    plt.show()
    return ret_img


# stddev = 0.#1 / ((1 + epoch) ** 0.99)  # the following two lines add gaussion noise to gradients to avoid the vanishing gradient problem (see Neelakantan et al. (2015)) AND clips gradients to avoid the exploding gradients problem.
# grad = [tf.clip_by_value(tf.add(gradient, tf.random.normal(stddev=stddev, mean=0., shape=gradient.shape)), -1., 1.) for gradient in grad]