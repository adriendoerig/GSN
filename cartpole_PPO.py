import tensorflow as tf
import numpy as np
import gym

def get_model_actor_simple(input_state_shape, n_actions):
    activation_fn = 'elu'
    output_activation_fn = 'softmax'
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_state_shape),
                                 # x,y,channels -> the point is that we don't take the "time" dimension of convLSTM
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(16, activation=activation_fn),
                                 tf.keras.layers.Dense(16, activation=activation_fn),
                                 tf.keras.layers.Dense(n_actions, activation=output_activation_fn)])
    model.summary()
    return model


def get_model_critic_simple(input_state_shape):
    activation_fn = 'elu'
    output_activation_fn = None
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_state_shape),
                                 # x,y,channels -> the point is that we don't take the "time" dimension of convLSTM
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(16, activation=activation_fn),
                                 tf.keras.layers.Dense(16, activation=activation_fn),
                                 tf.keras.layers.Dense(1, activation=output_activation_fn)])
    model.summary()
    return model


def get_advantages(values, rewards, masks, gamma, lmbda):
    returns = np.zeros_like(rewards)
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns[i] = gae + values[i]
    returns = returns.astype(np.float32)
    adv = returns - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


def ppo_loss_fn(oldpolicy_logprobs, newpolicy_logprobs, newpolicy_entropy, advantages, returns, new_values, clipping_val, alpha_critic_loss, alpha_entropy_loss):
    policies_logprob_diff = newpolicy_logprobs - oldpolicy_logprobs  # NEW [newpolicy_logprobs[i]-oldpolicy_logprobs[i] for i in range(len(newpolicy_logprobs))]
    ratio = tf.math.exp(policies_logprob_diff)
    p1 = ratio * advantages
    p2 = tf.clip_by_value(ratio, clip_value_min=1-clipping_val, clip_value_max=1+clipping_val) * advantages
    actor_loss = -tf.reduce_mean(tf.math.minimum(p1, p2))  # the PPO objective must be maximized, but we minimize loss -> flip sign

    critic_loss = alpha_critic_loss*tf.reduce_mean(tf.math.squared_difference(returns, new_values))
    entropy_loss = -alpha_entropy_loss*tf.reduce_mean(newpolicy_entropy)  # we want to maximize entropy -> flip sign
    total_loss = actor_loss + critic_loss + entropy_loss
    KL_divergence = tf.keras.losses.kullback_leibler_divergence(tf.exp(oldpolicy_logprobs), tf.exp(newpolicy_logprobs))
    return total_loss, actor_loss, critic_loss, entropy_loss, KL_divergence


def ppo_training_step(states_batch, actor_network, critic_network, optimizer, old_logprobs, returns, advantages, clipping_val, alpha_critic_loss, alpha_entropy_loss):
    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        # get new_actions, new_log_probs and new_entropies from actor. get new_values from critic
        action_probs = tf.squeeze(actor_network(states_batch))
        actions = tf.random.categorical(action_probs, 1) # NEW line
        indices = np.concatenate((np.expand_dims(np.arange(action_probs.shape[0]), axis=0).T, actions), axis=1)  # this line and the next are equivalent to new_probs = action_probs[actions] in numpy...
        new_probs = tf.expand_dims(tf.gather_nd(action_probs, indices), axis=1)
        new_logprobs = tf.math.log(new_probs + 1e-10)  # NEW tf.math.log(action_probs + 1e-10)
        new_entropies = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=-1)
        new_values = critic_network(states_batch)

        # compute ppo_loss based on the differences between these new values and the old ones collected while exploring the world
        ppo_loss, actor_loss, critic_loss, entropy_loss, KL_divergence = \
            ppo_loss_fn(old_logprobs, new_logprobs, new_entropies, returns, advantages, new_values,
                        clipping_val, alpha_critic_loss, alpha_entropy_loss)

        # update actor and critic networks
        actor_grad = actor_tape.gradient(ppo_loss, actor_network.trainable_variables)
        critic_grad = critic_tape.gradient(ppo_loss, critic_network.trainable_variables)
        optimizer.apply_gradients(zip(actor_grad, actor_network.trainable_variables))
        optimizer.apply_gradients(zip(critic_grad, critic_network.trainable_variables))
    return ppo_loss, actor_loss, critic_loss, entropy_loss, KL_divergence, actor_grad, critic_grad


def test_reward():
    state = env.reset()
    done = False
    total_reward = 0
    # print('testing...')
    limit = 0
    while not done:
        state_input = np.expand_dims(state, 0)
        action_probs = actor_network(state_input)
        action = np.argmax(action_probs)
        # value = tf.squeeze(critic_network(state_input))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        limit += 1
        if limit > 200:
            break
    return total_reward


########################################################################################################################
# MAIN
########################################################################################################################


do_training = 1

RUN_ID = 8

task = "CartPole-v1"
env = gym.make(task)

state = env.reset()
state_dims = env.observation_space.shape
n_actions = env.action_space.n

actor_network = get_model_actor_simple(state_dims, n_actions)
critic_network = get_model_critic_simple(state_dims)
# lr_decayed_fn = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps=100, decay_rate=0.9) if task == "CartPole-v1" else tf.keras.optimizers.schedules.ExponentialDecay(5e-3, decay_steps=100, decay_rate=0.9)
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

actor_ckpt_dict = {}
actor_ckpt_dict['ckpt'] = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=actor_network)
actor_ckpt_dict['ckpt_path'] = './CARTPOLE_model_checkpoints/RUN{}/actor_ckpt'.format(RUN_ID)
actor_ckpt_dict['manager'] = tf.train.CheckpointManager(actor_ckpt_dict['ckpt'], actor_ckpt_dict['ckpt_path'], max_to_keep=1)
critic_ckpt_dict = {}
critic_ckpt_dict['ckpt'] = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=critic_network)
critic_ckpt_dict['ckpt_path'] = './CARTPOLE_model_checkpoints/RUN{}/critic_ckpt'.format(RUN_ID)
critic_ckpt_dict['manager'] = tf.train.CheckpointManager(critic_ckpt_dict['ckpt'], critic_ckpt_dict['ckpt_path'], max_to_keep=1)

target_reached = False
best_reward = -1e5  # small
iters = 0
max_iters = 1000

ppo_buffer_size = 256 if task == "CartPole-v1" else 256
n_ppo_epochs = 2
ppo_batch_size = 64
n_ppo_batches = ppo_buffer_size//ppo_batch_size

gamma = 0.99  # discount factor (for ppo loss)
lmbda = 0.95  # lambda in ppo loss
clipping_val = 0.2 if task == "CartPole-v1" else 0.2  # clipping for ppo loss
alpha_critic_loss = 0.5 if task == "CartPole-v1" else 0.5  # weight of critic in total ppo loss
alpha_entropy_loss = 0.001 if task == "CartPole-v1" else 0.00001  # weight of entropy in total ppo loss

# tensorboard & gifs
log_dir = './CARTPOLE_tensorboard_logdir/'+task+'/RUN{}/'.format(RUN_ID)
summary_writer = tf.summary.create_file_writer(log_dir)

grad_step = 0  # counts number of gradient steps we took

if do_training:
    while not target_reached and iters < max_iters:

        # initialize variables to contain the experienced states, values, logprobs etc.
        # We will need timesteps+1 values for each image, to get the advantages of each states (need value(t+1))
        values = np.zeros((ppo_buffer_size + 1, 1))
        # all other variables have img_batch_size*timesteps values
        states = np.zeros(((ppo_buffer_size,) + state_dims))
        old_log_probs = np.zeros((ppo_buffer_size, 1))
        rewards = np.zeros((ppo_buffer_size, 1))
        returns = np.zeros((ppo_buffer_size, 1))
        entropies = np.zeros((ppo_buffer_size, 1))
        masks = np.zeros((ppo_buffer_size, 1))
        advantages = np.zeros((ppo_buffer_size, 1))

        for itr in range(ppo_buffer_size):
            states[itr] = state
            state_input = tf.expand_dims(state, 0)
            values[itr] = tf.squeeze(critic_network(state_input))

            action_probs = actor_network(state_input)
            action = tf.squeeze(tf.random.categorical(action_probs, 1))
            old_log_probs[itr] = tf.math.log(action_probs[0, action] + 1e-10)  # NEW tf.math.log(action_probs + 1e-10)
            observed_log_probs = tf.math.log(action_probs + 1e-10)  # NEW line
            entropies[itr] = -tf.reduce_sum(action_probs * observed_log_probs)  # NEW: didn't have observed_log_probs before

            state, rewards[itr], done, info = env.step(action.numpy())
            masks[itr] = not done

            if done:
                state = env.reset()

        values[-1] = tf.squeeze(critic_network(tf.expand_dims(state, 0)))

        # compute returns and advantages for the sequence from last image
        returns, advantages = get_advantages(values, rewards, masks, gamma, lmbda)

        for ppo_epoch in range(n_ppo_epochs):
            for batch in range(n_ppo_batches):
                ppo_loss, actor_loss, critic_loss, entropy_loss, KL_divergence, actor_grad, critic_grad = \
                    ppo_training_step(states[batch:batch+ppo_batch_size], actor_network, critic_network, optimizer,
                                      old_log_probs[batch:batch+ppo_batch_size],
                                      returns[batch:batch+ppo_batch_size], advantages[batch:batch+ppo_batch_size],
                                      clipping_val, alpha_critic_loss, alpha_entropy_loss)
                # if ppo_loss > 1e5:
                #     print('HUGE loss. ACTOR_GRADS (max {}, min {}), isNaN {}, CRITIC_GRADS (max {}, min {}), isNaN {})'.format
                #           (np.max(actor_grad[0]), np.min(actor_grad[0]), np.any(np.isnan(actor_grad[0])), np.max(critic_grad[0]), np.min(critic_grad[0]), np.any(np.isnan(critic_grad[0]))))
                grad_step += 1
            # shuffle data after each epoch
            idx = np.random.permutation(states.shape[0])
            states, old_log_probs, returns, advantages = states[idx], old_log_probs[idx], returns[idx], advantages[idx]

        n_tests = 5 if task == 'CartPole-v1' else 50
        avg_reward = np.mean([test_reward() for _ in range(n_tests)])
        if avg_reward > best_reward: best_reward = avg_reward
        print('\rIteration {}: total test reward = {}. best reward = {}. Last action_log_probs = {}, last action probs = {}'.format(iters, avg_reward, best_reward, old_log_probs[-1], action_probs[-1]), end='')

        iters += 1
        env.reset()

        if iters%1 == 0:
            with summary_writer.as_default():
                # tf.summary.scalar('__LR', optimizer.lr(grad_step), step=iters)
                tf.summary.scalar('0_TESTING_avg_reward', avg_reward, step=iters)
                tf.summary.scalar('1a_OBSERVATIONS_avg_advantage', tf.squeeze(advantages[-1]), step=iters)
                tf.summary.scalar('1b_OBSERVATIONS_%reward', tf.reduce_mean(rewards)*ppo_buffer_size/100, step=iters)
                tf.summary.scalar('1c_OBSERVATIONS_avg_predicted_value', tf.reduce_mean(values), step=iters)
                tf.summary.scalar('2_PPO_loss', ppo_loss, step=iters)
                tf.summary.scalar('3_actor_loss', actor_loss, step=iters)
                tf.summary.scalar('4_critic_loss', critic_loss, step=iters)
                tf.summary.scalar('5_entropy_loss', entropy_loss, step=iters)
                tf.summary.scalar('6_OBSERVATIONS_entropies', tf.reduce_mean(entropies), step=iters)
                tf.summary.scalar('7_KL_divergence', tf.reduce_mean(KL_divergence), step=iters)
                tf.summary.histogram('OBSERVATIONS_log_probs', observed_log_probs, step=iters)  # NEW: didn't have observed_log_probs before

        criterion = 195 if task == 'CartPole-v1' else 0.9
        if best_reward > criterion or iters > max_iters:
            target_reached = True
            actor_ckpt_dict['manager'].save()
            critic_ckpt_dict['manager'].save()

# show some behaviour at the end
print('\nVisualizing behaviour of trained networks.')
actor_ckpt_dict['ckpt'].restore(actor_ckpt_dict['manager'].latest_checkpoint)
critic_ckpt_dict['ckpt'].restore(critic_ckpt_dict['manager'].latest_checkpoint)
state = env.reset()
env.render()
max_its = 600
finished = 0
for it in range(max_its):
    state_input = np.expand_dims(state, 0)
    action_probs = actor_network(state_input)
    action = np.argmax(action_probs)
    state, _, done, _ = env.step(action)
    env.render()
    if (done or it == max_its-1) and not finished:
        print('Lasted {} steps.'.format(it if it < 499 else 'the maximal number of'))
        finished = 1

env.close()