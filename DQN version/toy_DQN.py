import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from helper_functions import make_base_model, make_toy_dataset, train_on_toy_dataset, get_retina_pars, plot_retinal_image


# Dataset creation
train_images, train_labels, test_images, test_labels, finetuning_train_images, finetuning_train_labels, finetuning_test_images, finetuning_test_labels = make_toy_dataset()
img_shape = train_images[0].shape
batch_size = 64
n_epochs = 1

# experiment with the retina functions
# test_img = np.repeat(np.vstack([np.hstack(train_images[:4]), np.hstack(train_images[4:8]), np.hstack(train_images[8:12]), np.hstack(train_images[12:16])]), 3, axis=-1)
# in_size = 28
# out_size = 28
# retina_pars = get_retina_pars(in_size, out_size)
# retinal_image = plot_retinal_image(train_images[0,:, :, 0], in_size, out_size)

# loss & optimizer we will use
classification_loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.optimizers.Adam()

# define base CNN model and load if checkpoint exists, otherwise train
n_LSTM_units = 10
n_frames = 8
n_output_units = 10
LSTM_model, ckpt, ckpt_path, manager = make_base_model(img_shape, n_LSTM_units, n_frames, n_output_units, optimizer, 'basic_LSTM')
train_on_toy_dataset(LSTM_model, n_frames, train_images, train_labels, test_images, test_labels, batch_size, n_epochs, classification_loss, optimizer, ckpt, ckpt_path, manager)
