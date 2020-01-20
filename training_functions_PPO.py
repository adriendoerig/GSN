import tensorflow as tf
import numpy as np
import imgaug.augmenters as iaa
import os
from toy_dataset_functions_PPO import denormalize_img, save_sequence_gif, plot_fix_heatmap, plot_fix_path


def get_advantages(values, rewards, gamma, lmbda):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * lmbda * gae
        returns.insert(0, gae + values[i])
    adv = tf.convert_to_tensor(returns) - values[:-1]
    return returns, (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-10)


def ppo_loss_fn(oldpolicy_logprobs, newpolicy_logprobs, newpolicy_entropy, advantages, returns, new_values, clipping_val, alpha_critic_loss, alpha_entropy_loss):
    policies_logprob_diff = [newpolicy_logprobs[i]-oldpolicy_logprobs[i] for i in range(len(newpolicy_logprobs))]
    ratio = tf.reduce_mean(tf.math.exp(policies_logprob_diff), axis=-1)
    p1 = ratio * advantages
    p2 = tf.clip_by_value(ratio, clip_value_min=1 - clipping_val, clip_value_max=1 + clipping_val) * advantages
    actor_loss = -tf.reduce_mean(tf.math.minimum(p1, p2))  # the PPO objective must be maximized, but we minimize loss -> flip sign
    critic_loss = alpha_critic_loss*tf.reduce_mean(tf.math.squared_difference(returns, new_values))
    entropy_loss = -alpha_entropy_loss*tf.reduce_mean(newpolicy_entropy)  # we want to maximize entropy -> flip sign
    total_loss = actor_loss + critic_loss + entropy_loss
    return total_loss, actor_loss, critic_loss, entropy_loss


def ppo_training_step(states_batch, saccade_generator, old_logprobs, returns, advantages, clipping_val, alpha_critic_loss, alpha_entropy_loss):
    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        # get new_actions, new_log_probs and new_entropies from actor. get new_values from critic
        new_actions, new_logprobs, new_entropies = saccade_generator.get_action_log_prob_and_entropy(states_batch)
        new_values = saccade_generator.get_value(states_batch)

        # compute ppo_loss based on the differences between these new values and the old ones collected while exploring the world
        ppo_loss, actor_loss, critic_loss, entropy_loss = \
            ppo_loss_fn(old_logprobs, new_logprobs, new_entropies, returns, advantages, new_values,
                        clipping_val, alpha_critic_loss, alpha_entropy_loss)

        # update actor and critic networks
        sac_actor_grad = actor_tape.gradient(ppo_loss, saccade_generator.actor_network.trainable_variables)
        sac_critic_grad = critic_tape.gradient(ppo_loss, saccade_generator.critic_network.trainable_variables)
        saccade_generator.optimizer.apply_gradients(zip(sac_actor_grad, saccade_generator.actor_network.trainable_variables))
        saccade_generator.optimizer.apply_gradients(zip(sac_critic_grad, saccade_generator.critic_network.trainable_variables))
    return ppo_loss, actor_loss, critic_loss, entropy_loss



def train_GSNs(visual_system, saccade_generator, readout_layer, timesteps, init_fix_pos,
               train_images, train_labels, train_mean, train_std, data_augment,
               n_img_epochs, img_batch_size, n_ppo_epochs, ppo_buffer_size, ppo_batch_size, only_last_loss, binary_reward,
               gamma, lmbda, clipping_val, alpha_critic_loss, alpha_entropy_loss,
               visual_system_ckpt_dict, saccade_actor_ckpt_dict, saccade_critic_ckpt_dict, RUN_ID):

    # load networks if they exist
    if os.path.exists(visual_system_ckpt_dict['ckpt_path']):
        print('Loading visual system')
        visual_system_ckpt_dict['ckpt'].restore(visual_system_ckpt_dict['manager'].latest_checkpoint)
    if os.path.exists(saccade_actor_ckpt_dict['ckpt_path']):
        print('Loading saccade actor')
        saccade_actor_ckpt_dict['ckpt'].restore(saccade_actor_ckpt_dict['manager'].latest_checkpoint)
    if os.path.exists(saccade_critic_ckpt_dict['ckpt_path']):
        print('Loading saccade critic')
        saccade_critic_ckpt_dict['ckpt'].restore(saccade_critic_ckpt_dict['manager'].latest_checkpoint)

    if data_augment: augment_seq = iaa.Sequential([iaa.OneOf([iaa.AdditiveGaussianNoise(scale=0.03), iaa.AdditiveLaplaceNoise(scale=0.03), iaa.Dropout(0.03)])], random_order=True)

    # tensorboard & gifs
    log_dir = './tensorboard_logdir/' + RUN_ID
    summary_writer = tf.summary.create_file_writer(log_dir)
    tf.summary.trace_on(graph=True, profiler=False)
    save_gif_cond = 100  # save gifs every X batches
    save_gif_path = './outputs/' + RUN_ID

    # initializations
    n_img_batches = train_images.shape[0] // img_batch_size  # there are n_img_batches batches of images in one epoch of MNIST
    n_ppo_batches = ppo_buffer_size*timesteps // ppo_batch_size  # we do the PPO update after each img_batch, and it has img_batch_size*timesteps states
    state_shape = visual_system.network.get_layer(readout_layer).output_shape
    ppo_loss, actor_loss, critic_loss, entropy_loss, RL_mean_reward = 0., 0., 0., 0., 0.  # need to be defined for tensorboard logs prior to the first ppo update

    # initialize variables to contain the experienced states, values, logprobs etc.
    # We will need timesteps+1 values for each image, to get the advantages of each states (need value(t+1))
    values = np.zeros((ppo_buffer_size * (timesteps + 1), 1))
    values_counter = 0
    # all other variables have img_batch_size*timesteps values
    states = np.zeros(((ppo_buffer_size * timesteps,) + state_shape))
    old_log_probs = np.zeros((ppo_buffer_size * timesteps, saccade_generator.n_actions))
    rewards = np.zeros((ppo_buffer_size * timesteps, 1))
    returns = np.zeros((ppo_buffer_size * timesteps, 1))
    advantages = np.zeros((ppo_buffer_size * timesteps, 1))
    states_counter = 0

    for epoch in range(n_img_epochs):
        for n in range(n_img_batches):
            with tf.GradientTape() as vis_sys_tape:

                visual_system_loss = 0.
                batch_accuracy = np.zeros((timesteps,))

                for batch_img in range(img_batch_size):

                    if n%save_gif_cond == 0 and batch_img <= 5:
                        input_seq_for_gif = []
                        state_seq_for_gif = []
                        fix_seq = []
                        states_var_seq = []

                    img, label = train_images[n*img_batch_size + batch_img], train_labels[n*img_batch_size + batch_img]
                    if data_augment: img = augment_seq(images=img)

                    visual_system.set_fix_pos(init_fix_pos())

                    for t in range(timesteps):

                        # get the current state
                        this_state = visual_system.state if visual_system.state is not None else np.random.normal(0, 1, (state_shape))  # Store current state
                        states[states_counter] = this_state  # initial state
                        value = saccade_generator.get_value(this_state)
                        values[values_counter] = value

                        # take action and get associated log_probs, get next state
                        action, old_log_prob, entropy = saccade_generator.get_action_log_prob_and_entropy(this_state)  # Query actor for the next action
                        prediction, next_state, accurate, pred_loss, center_bias_loss, next_img, next_fix_pos = visual_system.take_action(img, label, action, train_mean, train_std)  # Take action, get new state and reward (the mean and std are for normalizing)
                        batch_accuracy[t] += accurate/img_batch_size
                        old_log_probs[states_counter] = tf.squeeze(old_log_prob)

                        # add the predictino loss to the visual system loss if the time is right
                        if (only_last_loss and t == timesteps-1) or (not only_last_loss):
                            visual_system_loss += pred_loss

                        # get the adequate reward, depending on the setup (e.g. binary reward, only_last_loss, etc.)
                        # if only_last_loss: reward = 0 except for last step.
                        if (only_last_loss and t == timesteps-1) or (not only_last_loss):
                            rewards[states_counter] = accurate if binary_reward else -pred_loss - center_bias_loss  # if binary reward: 1 or 0. else: use visual system loss as reward. loss is minimized, but reward is maximized -> flip sign
                        else:
                            rewards[states_counter] = 0

                        # collect frames to save gifs every few batches
                        if n%save_gif_cond == 0 and batch_img <= 5:
                            input_seq_for_gif.append(denormalize_img(next_img[0, :, :, :], train_mean, train_std))  #
                            state_seq_for_gif.append(next_state[0])  #
                            fix_seq.append(next_fix_pos)
                            states_var_seq.append(np.std(next_state[0]))

                        states_counter += 1
                        values_counter += 1

                    # get value of last state (needed to compute the advantage)
                    values[values_counter] = saccade_generator.get_value(next_state)
                    values_counter += 1

                    # compute returns and advantages for the sequence from last image
                    returns[states_counter-timesteps:states_counter], advantages[states_counter-timesteps:states_counter] \
                        = get_advantages(values[values_counter-(timesteps+1):values_counter], rewards[states_counter-timesteps:states_counter], gamma, lmbda)

                    # reset the convlstm states since we have finished processing this image
                    visual_system.network.reset_states()

                    # save gifs every few batches
                    if n%save_gif_cond == 0 and batch_img <= 5:
                            gif_name = '/epoch_{}_batch_{}_img_{}'.format(epoch,n,batch_img)
                            print('saving sequence gif to '+save_gif_path+gif_name)
                            print('fixation sequence(t): {}'.format([fix for fix in fix_seq]))
                            print('state std sequence(t): {}'.format(states_var_seq))
                            save_sequence_gif(tf.stack(tf.cast(input_seq_for_gif, tf.float32)), path=save_gif_path, state_sequence=state_seq_for_gif, name=gif_name)

                # train the visual system for one step on loss accumulated over time
                grad = vis_sys_tape.gradient(visual_system_loss, visual_system.network.trainable_variables)
                visual_system.optimizer.apply_gradients(zip(grad, visual_system.network.trainable_variables))
                visual_system_ckpt_dict['ckpt'].step.assign_add(1)

                # train actor and critic using PPO
                if states_counter == (ppo_buffer_size)*timesteps:
                    RL_mean_reward = timesteps * tf.reduce_mean(rewards) if only_last_loss else tf.reduce_mean(rewards)
                    for ppo_epoch in range(n_ppo_epochs):
                        for batch in range(n_ppo_batches):
                            ppo_loss, actor_loss, critic_loss, entropy_loss = \
                                ppo_training_step(states[batch:batch+ppo_batch_size], saccade_generator,
                                                  old_log_probs[batch:batch+ppo_batch_size],
                                                  returns[batch:batch+ppo_batch_size], advantages[batch:batch+ppo_batch_size],
                                                  clipping_val, alpha_critic_loss, alpha_entropy_loss)
                            saccade_actor_ckpt_dict['ckpt'].step.assign_add(1)
                            saccade_critic_ckpt_dict['ckpt'].step.assign_add(1)
                    states_counter, values_counter = 0, 0  # we will start a new ppo_buffer and overwrite this one

                print('Epoch {} batch: {}, visual_system_loss = {}, previous RL_mean_reward = {}, accuracy(t) = {}.'.format(epoch, n, visual_system_loss, RL_mean_reward, batch_accuracy))
                if n % 10 == 0:
                    # save data to tensorboard.
                    with summary_writer.as_default():
                        vis_step = tf.cast(visual_system_ckpt_dict['ckpt'].step, tf.int64)
                        sac_step = tf.cast(saccade_actor_ckpt_dict['ckpt'].step, tf.int64)
                        tf.summary.scalar('0_class_lr', visual_system.optimizer.lr(visual_system_ckpt_dict['ckpt'].step), step=vis_step)
                        tf.summary.scalar('1_RL_lr', saccade_generator.optimizer.lr(sac_step), step=sac_step)
                        tf.summary.scalar('2_visual_system_loss', visual_system_loss, step=vis_step)
                        tf.summary.scalar('3a_PPO_loss', ppo_loss, step=sac_step)
                        tf.summary.scalar('3b_actor_loss', actor_loss, step=sac_step)
                        tf.summary.scalar('3c_critic_loss', critic_loss, step=sac_step)
                        tf.summary.scalar('3d_entropy_loss', entropy_loss, step=sac_step)
                        tf.summary.scalar('3dprime_entropy_last frame', tf.squeeze(entropy), step=sac_step)
                        tf.summary.scalar('4_mean_reward', RL_mean_reward, step=sac_step)
                        if visual_system.action_coding == 'continuous':
                            tf.summary.scalar('5a_action_x', tf.squeeze(action)[1], step=sac_step)
                            tf.summary.scalar('5b_action_y', tf.squeeze(action)[0], step=sac_step)
                        else:
                            tf.summary.scalar('5_action', tf.squeeze(action), step=sac_step)
                        vis_layer_0_weights = visual_system.network.get_layer('conv2d').get_weights()
                        vis_layer_readout_weights = visual_system.network.get_layer(readout_layer).get_weights()
                        sac_actor_layer_0_weights = saccade_generator.actor_network.layers[1].get_weights()  # 0 if conv sec_net, 1 if fully_connected
                        sac_actor_layer_1_weights = saccade_generator.actor_network.layers[-1].get_weights()  # -3 if conv sec_net, -1 if fully_connected
                        sac_critic_layer_0_weights = saccade_generator.critic_network.layers[1].get_weights()  # 0 if conv sec_net, 1 if fully_connected
                        sac_critic_layer_1_weights = saccade_generator.critic_network.layers[-1].get_weights()  # -3 if conv sec_net, -1 if fully_connected
                        tf.summary.histogram('6a_vis_system_weights_1ker', vis_layer_0_weights[0], step=vis_step)
                        tf.summary.histogram('6b_vis_system_weights_1bias', vis_layer_0_weights[1], step=vis_step)
                        tf.summary.histogram('6c_vis_system_weights_readout0', vis_layer_readout_weights[0], step=vis_step)
                        tf.summary.histogram('6d_vis_system_weights_readout1', vis_layer_readout_weights[1], step=vis_step)
                        tf.summary.histogram('7a_sac_actor_weights_0ker', sac_actor_layer_0_weights[0], step=sac_step)
                        tf.summary.histogram('7b_sac_actor_weights_0bias', sac_actor_layer_0_weights[1], step=sac_step)
                        tf.summary.histogram('7c_sac_actor_weights_1ker', sac_actor_layer_1_weights[0], step=sac_step)
                        tf.summary.histogram('7d_sac_actor_weights_1bias', sac_actor_layer_1_weights[1], step=sac_step)
                        tf.summary.histogram('7e_sac_critic_weights_0ker', sac_critic_layer_0_weights[0], step=sac_step)
                        tf.summary.histogram('7f_sac_critic_weights_0bias', sac_critic_layer_0_weights[1], step=sac_step)
                        tf.summary.histogram('7g_sac_critic_weights_1ker', sac_critic_layer_1_weights[0], step=sac_step)
                        tf.summary.histogram('7h_sac_critic_weights_1bias', sac_critic_layer_1_weights[1], step=sac_step)
                        tf.summary.histogram('8a_log_probs_frame0', old_log_probs[0], step=sac_step)
                        tf.summary.histogram('8b_log_probs_frame1', old_log_probs[1], step=sac_step)
                        tf.summary.histogram('8c_log_probs_frame2', old_log_probs[2], step=sac_step)
                        tf.summary.histogram('8d_log_probs_frame3', old_log_probs[3], step=sac_step)
                        tf.summary.histogram('8e_log_probs_frame4', old_log_probs[4], step=sac_step)
                        tf.summary.histogram('8f_log_probs_frame5', old_log_probs[5], step=sac_step)

                if n % 200 == 0 and n!=0:
                    print('SAVING MODELS TO {}, {} and {}'.format(visual_system_ckpt_dict['ckpt_path'], saccade_actor_ckpt_dict['ckpt_path'], saccade_critic_ckpt_dict['ckpt_path']))
                    visual_system_ckpt_dict['manager'].save()
                    saccade_actor_ckpt_dict['manager'].save()
                    saccade_critic_ckpt_dict['manager'].save()
                    print('Plotting fixation heatmaps and paths')
                    plot_fix_heatmap(visual_system, saccade_generator, timesteps, state_shape, init_fix_pos, train_images[:1000], train_labels[:1000], train_mean, train_std, save_gif_path+'/_fix_heatmaps_training_step_{}.png'.format(sac_step))
                    plot_fix_path(visual_system, saccade_generator, timesteps, state_shape, init_fix_pos, train_images[:25], train_labels[:25], train_mean, train_std, save_gif_path+'/_fix_paths_training_step_{}.png'.format(sac_step))


def test_GSNs(random_fixations, visual_system, saccade_generator, readout_layer, timesteps, init_fix_pos,
              test_images, test_labels, train_mean, train_std,
              visual_system_ckpt_dict, saccade_actor_ckpt_dict, saccade_critic_ckpt_dict,
              only_last_loss, binary_reward, RUN_ID):

    if os.path.exists(visual_system_ckpt_dict['ckpt_path']):
        print('Loading visual system')
        visual_system_ckpt_dict['ckpt'].restore(visual_system_ckpt_dict['manager'].latest_checkpoint)
    if os.path.exists(saccade_actor_ckpt_dict['ckpt_path']):
        print('Loading saccade actor')
        saccade_actor_ckpt_dict['ckpt'].restore(saccade_actor_ckpt_dict['manager'].latest_checkpoint)
    if os.path.exists(saccade_critic_ckpt_dict['ckpt_path']):
        print('Loading saccade critic')
        saccade_critic_ckpt_dict['ckpt'].restore(saccade_critic_ckpt_dict['manager'].latest_checkpoint)

    save_gif_path = './sequence_gifs/' + RUN_ID
    state_shape = visual_system.network.get_layer(readout_layer).output_shape  # shape of this layer
    n_test_stimuli = 100
    test_accuracy = np.zeros((timesteps,))

    for test_img in range(n_test_stimuli):

        img, label = test_images[test_img], test_labels[test_img]
        input_seq_for_gif = []
        state_seq_for_gif = []
        fix_seq = []
        states_var_seq = []
        visual_system_loss, test_mean_reward = 0., 0.
        visual_system.set_fix_pos(init_fix_pos())
        states, rewards, new_log_probs, entropies, values = [], [], [], [], []

        for t in range(timesteps):

            # get old state
            old_state = visual_system.state if visual_system.state is not None else np.random.normal(0, .1, (
                state_shape))  # Store current state
            if t == 0: states.append(old_state)  # initial state
            value = saccade_generator.get_value(old_state)
            values.append(value)

            # take action and get new state, etc...
            action, new_log_prob, entropy = saccade_generator.get_action_log_prob_and_entropy(
                old_state)  # Query actor for the next action
            prediction, new_state, accurate, pred_loss, center_bias_loss, new_img, new_fix_pos = visual_system.take_action(img, label, action, train_mean, train_std)  # Take action, get new state and reward (the mean and std are for normalizing)
            test_accuracy[t] += accurate / n_test_stimuli
            new_log_probs.append(tf.squeeze(new_log_prob))
            entropies.append(entropy)

            # add the predictino loss to the visual system loss if the time is right
            if (only_last_loss and t == timesteps - 1) or (not only_last_loss):
                visual_system_loss += pred_loss

            # get the adequate reward, depending on the setup (e.g. binary reward, only_last_loss, etc.)
            # if only_last_loss: reward = 0 except for last step.
            if only_last_loss and t < timesteps - 1:
                reward = accurate if binary_reward else -pred_loss - center_bias_loss  # if binary reward: 1 or 0. else: use visual system loss as reward. loss is minimized, but reward is maximized -> flip sign
            else:
                reward = 0
            rewards.append(reward)
            test_mean_reward += reward / (timesteps * n_test_stimuli)

            if test_img <= 10:
                input_seq_for_gif.append(denormalize_img(new_img[0, :, :, :], train_mean, train_std))  #
                state_seq_for_gif.append(new_state[0])  #
                fix_seq.append(new_fix_pos)
                states_var_seq.append(np.std(new_state[0]))
            if test_img == n_test_stimuli - 1 and t == timesteps - 1:  # Print out progress every few iterations
                print('Testing: pred_loss = {}, accuracy(t) = {}. reward = {}'.format(visual_system_loss, test_accuracy, test_mean_reward))

        visual_system.network.reset_states()  # resets the convlstm states since we have finished processing this image

        if test_img <= 10:
            gif_name = '/TESTING_random_fix_{}_img_{}'.format(random_fixations, test_img)
            print('saving sequence gif to ' + save_gif_path + gif_name)
            print('fixation sequence(t): {}'.format([fix for fix in fix_seq]))
            print('state std sequence(t): {}'.format(states_var_seq))
            save_sequence_gif(tf.stack(tf.cast(input_seq_for_gif, tf.float32)), path=save_gif_path, state_sequence=state_seq_for_gif, name=gif_name)

    plot_fix_heatmap(visual_system, saccade_generator, timesteps, state_shape, init_fix_pos, test_images, test_labels, train_mean, train_std, save_gif_path+'/_fix_heatmaps_test_random_{}.png'.format(str(random_fixations)))
    plot_fix_path(visual_system, saccade_generator, timesteps, state_shape, init_fix_pos, test_images[:16], test_labels[:16], train_mean, train_std, save_gif_path+'/_fix_paths_test_random_{}.png'.format(str(random_fixations)))

    return test_accuracy
