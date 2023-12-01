import tensorflow as tf
import numpy as np

from tqdm import tqdm
from functools import partial

from . import networks 
from . import buffer 


class DDPGAgent(tf.Module):

    def __init__(
        self,
        state_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        action_bounds: list[tuple[float, float]],
        twin_delayed: bool = False,
        units: int = 256,
        activation: str = 'elu',
        dropout: float = None,
        batch_norm: bool = False,
        batch_size: int = 64,
        initial_learning_rate: float = 1e-5,
        learning_rate_decay_steps: int = 100_000,
        end_learning_rate: float = 1e-6,
        update_after: int = 0,
        buffer_size: int = 50_000,
        gamma: float = 0.99,
        tau: float = 0.005,
        initial_actor_noise: float = 0.1,
        noise_decay_steps: int = 10_000,
        end_actor_noise: float = 0.01,
        target_actor_noise: float = 0.2,
        policy_delay: int = 1,
        target_delay: int = 1,
        start_steps: int = 1_000,
        save_path: str = None,
        name: str = 'DDPGAgent'
    ):
        super().__init__(name=name)

        PolicyNetwork = partial(
            networks.PolicyNetwork, 
            state_shape=state_shape,
            action_bounds=action_bounds,
            units=units,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

        ValueNetwork = partial(
            networks.ValueNetwork, 
            state_shape=state_shape,
            action_shape=action_shape,
            units=units,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

        self.save_path = save_path
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.twin_delayed = twin_delayed
        self.update_after = max(batch_size, update_after)
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = 1 if not self.twin_delayed else max(policy_delay, 1)
        self.target_delay = target_delay
        self.start_steps = start_steps

        self.buffer = buffer.Buffer(buffer_size, batch_size)

        self.policy = PolicyNetwork()
        self.target_policy = PolicyNetwork()
        self.target_policy.set_weights(self.policy.get_weights())

        self.value = ValueNetwork()
        self.target_value = ValueNetwork()
        self.target_value.set_weights(self.value.get_weights())

        if self.twin_delayed:
            self.value_2 = ValueNetwork()
            self.target_value_2 = ValueNetwork()
            self.target_value_2.set_weights(self.value_2.get_weights())

        actor_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=learning_rate_decay_steps//self.policy_delay,
            end_learning_rate=end_learning_rate,
            power=2.0
        )
        self.optimizer_actor = tf.keras.optimizers.Adam(actor_scheduler)

        critic_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate * 2,
            decay_steps=learning_rate_decay_steps,
            end_learning_rate=end_learning_rate * 2,
            power=1.0)
        
        self.optimizer_critic = tf.keras.optimizers.Adam(critic_scheduler)

        if self.twin_delayed:
            critic_scheduler_2 = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=initial_learning_rate * 2,
                decay_steps=learning_rate_decay_steps,
                end_learning_rate=end_learning_rate * 2,
                power=1.0)
            self.optimizer_critic_2 = tf.keras.optimizers.Adam(critic_scheduler_2)

        self.action_bounds = list(map(tf.convert_to_tensor, zip(*action_bounds)))
        
        self.apply_action_noise = ActionNoise(
            shape=action_shape, 
            bounds=self.action_bounds, 
            initial_noise=initial_actor_noise, 
            end_noise=end_actor_noise, 
            decay_steps=noise_decay_steps)
                    
        if self.twin_delayed:
            self.apply_target_action_noise = TargetActionNoise(
                shape=action_shape, 
                bounds=self.action_bounds, 
                noise=target_actor_noise)

    def train(self, env, num_episodes: int) -> None:
            
        best_reward = float('-inf')
        reward_list = []

        for i in (pbar := tqdm(range(num_episodes), mininterval=0.025)):

            state, _ = env.reset()

            episodic_reward = 0.0

            while True:
                
                if i < self.start_steps:
                    action = tf.random.uniform(
                        self.action_shape, 
                        minval=self.action_bounds[0], 
                        maxval=self.action_bounds[1])
                else:
                    action = self(state, apply_noise=True)
                    
                next_state, reward, done, truncated, info = env.step(action)

                terminal = done or truncated 

                episodic_reward += reward

                done = np.array(terminal, np.float32)
                reward = np.array(reward, np.float32)
                action = np.array(info.get('action', action), np.float32)

                transition = (state, action, reward, next_state, done)

                self.buffer.add(transition)

                if len(self.buffer) >= self.update_after:

                    states, actions, rewards, next_states, dones = self.buffer.sample()

                    self.update_critic(states, actions, rewards, next_states, dones)

                    if i % self.policy_delay == 0:
                        self.update_actor(states)

                    if i % self.target_delay == 0:
                        self.update_actor_critic_target()

                if terminal:
                    break

                state = next_state.copy()

            reward_list.append(episodic_reward)

            if len(reward_list) > 10:
                reward_list.pop(0)

            running_average_reward = np.mean(reward_list)
            
            if hasattr(env, 'pbar_description'):
                pbar.set_description(env.pbar_description, refresh=False)
            else:
                pbar.set_description(f'Episode reward: {running_average_reward:.2f}')

            if i > num_episodes//100 and running_average_reward > best_reward:
               best_reward = running_average_reward
               best_weights = self.policy.get_weights()
               # tf.saved_model.save(self, self.save_path + f'current_policy')

        if isinstance(self.save_path, str):
            tf.saved_model.save(self, self.save_path + f'_policy')
            self.policy.set_weights(best_weights)
            tf.saved_model.save(self, self.save_path + f'_best_policy')

        pbar.close()

    @tf.function
    def update_critic(
        self, 
        states: tf.Tensor, 
        actions: tf.Tensor, 
        rewards: tf.Tensor, 
        next_states: tf.Tensor, 
        dones: tf.Tensor
    ) -> None:

        target_actions = self.target_policy(
            next_states, training=True)
        
        if self.twin_delayed:
            target_actions = self.apply_target_action_noise(target_actions)

        target_value = self.target_value(
            [next_states, target_actions], training=True)

        if self.twin_delayed:
            target_value_2 = self.target_value_2(
                [next_states, target_actions], training=True)
            
            target_value = tf.minimum(target_value, target_value_2)

        target_value = (
            tf.expand_dims(tf.cast(rewards, dtype=target_value.dtype), -1) +
            self.gamma *
            target_value *
            (1 - tf.expand_dims(tf.cast(dones, dtype=target_value.dtype), -1))
        )

        with tf.GradientTape() as tape:
            value = self.value([states, actions], training=True)
            loss = tf.reduce_mean(
                tf.math.squared_difference(target_value, value))

        gradients = tape.gradient(loss, self.value.trainable_weights)
        self.optimizer_critic.apply_gradients(
            zip(gradients, self.value.trainable_weights))

        if self.twin_delayed:
            with tf.GradientTape() as tape:
                value = self.value_2(
                    [states, actions], training=True)
                loss = tf.reduce_mean(
                    tf.math.squared_difference(target_value, value))

            gradients = tape.gradient(loss, self.value_2.trainable_weights)
            self.optimizer_critic_2.apply_gradients(
                zip(gradients, self.value_2.trainable_weights))

    @tf.function
    def update_actor(self, states: tf.Tensor) -> None:

        with tf.GradientTape() as tape:

            actions = self.policy(states, training=True)
            
            value = self.value([states, actions], training=True)

            loss = -tf.reduce_mean(value)

        gradients = tape.gradient(loss, self.policy.trainable_weights)
        self.optimizer_actor.apply_gradients(
            zip(gradients, self.policy.trainable_weights))

    @tf.function
    def update_actor_critic_target(self) -> None:

        def update(
            target_weights: list[tf.Variable], 
            weights: list[tf.Variable]
        ) -> None:
            for tw, w in zip(target_weights, weights):
                tw.assign(w * self.tau + tw * (1 - self.tau))

        update(self.target_policy.trainable_weights, 
               self.policy.trainable_weights)
        
        update(self.target_value.trainable_weights, 
               self.value.trainable_weights)

        if self.twin_delayed:
            update(self.target_value_2.trainable_weights, 
                   self.value_2.trainable_weights)

    @tf.function(
        reduce_retracing=True, 
        input_signature=[
            tf.TensorSpec((None,), tf.float32), 
            tf.TensorSpec((), tf.bool)
        ]
    )
    def __call__(
        self, 
        state: tf.Tensor, 
        apply_noise: bool = False
    ) -> tf.Tensor:
        action = tf.squeeze(self.policy(tf.expand_dims(state, 0)), 0)
        if apply_noise:
            action = self.apply_action_noise(action)
        return action
    

class ActionNoise(tf.Module):

    def __init__(self, shape, bounds, initial_noise, end_noise, decay_steps):
        super().__init__()
        self._step = tf.Variable(0.0, dtype=tf.float32, trainable=False) 
        self._shape = shape
        self._low_bounds = tf.convert_to_tensor(bounds[0], tf.float32)
        self._high_bounds = tf.convert_to_tensor(bounds[1], tf.float32)
        self._initial = tf.convert_to_tensor(initial_noise, tf.float32)
        self._end = tf.convert_to_tensor(end_noise, tf.float32) 
        self._decay_steps = tf.convert_to_tensor(decay_steps, tf.float32)
    
    def __call__(self, action):
        stddev = (
            (self._initial - self._end) *
            (1 - tf.minimum(self._step, self._decay_steps) / self._decay_steps) ** 2 +
            self._end
        )
        self._step.assign_add(1.0)
        return action + (
            self._random_truncated_normal(
                shape=self._shape, 
                loc=0.0, 
                scale=(stddev * self._high_bounds), 
                minimum=self._low_bounds, 
                maximum=self._high_bounds
            )
        )

    @staticmethod 
    def _random_truncated_normal(shape, loc, scale, minimum, maximum):
        values = tf.zeros(shape)
        logical_test = tf.cast(tf.zeros_like(values), dtype=tf.bool)
        while not tf.math.reduce_all(logical_test):
            logical_test = tf.logical_and(
                (minimum < values), (values < maximum))
            values = tf.where(
                logical_test, values, tf.random.normal(shape, loc, scale))
        return values


class TargetActionNoise(tf.Module):

    def __init__(self, shape, bounds, noise):
        super().__init__()
        self._shape = shape
        self._low_bounds = tf.convert_to_tensor(bounds[0], tf.float32)
        self._high_bounds = tf.convert_to_tensor(bounds[1], tf.float32)
        self._noise = tf.convert_to_tensor(noise, tf.float32)

    def __call__(self, action):
        return tf.clip_by_value(
            action + tf.random.truncated_normal(
                self._shape, 0.0, self._noise * self._high_bounds
            ),
            self._low_bounds, self._high_bounds)
            

