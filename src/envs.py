import tensorflow as tf
import numpy as np
import gymnasium as gym

from . import utils


class ScoutingRuns(gym.Env):

    _PHI_RANGE = (0.05, 0.95)
    _RT_NOISE = 0.002 # Inherent system noise obtained from real experiments, may vary
    _VOID_TIME = 0.258 # values obtained from real experiments, may vary
    _DWELL_TIME = 0.614 # values obtained from real experiments, may vary
    
    def __init__(
        self,
        max_scouting_runs: int = 3,
        penalty: float = 0.1,
        enforce_constraints: bool = False,
        stop_action: bool = False,
        tf_summary_path: str = None
    ):
        super().__init__()

        self.enforce_constraints = enforce_constraints
        self.stop_action = stop_action
        self.penalty = penalty
        self.max_scouting_runs = max_scouting_runs or 3
        
        min_t_gradient = 0.1 if not self.enforce_constraints else 1.0

        # gradient duration is in log scale: t'_{G} = log(t_{G} + 1)
        # maximum gradient duration is approximately 600 minutes, which
        # translates to 6.4 log-minutes:
        max_t_gradient = 6.4 

        # phi_end is phi_incr if enforcing contraints
        min_phi_end = 0.00 if self.enforce_constraints else 0.05 

        if self.stop_action:
            self.action_space = gym.spaces.Box(
                low=np.array([0.00, self._PHI_RANGE[0], min_phi_end, min_t_gradient]),
                high=np.array([1.00, self._PHI_RANGE[1], self._PHI_RANGE[1], max_t_gradient]),
                shape=(4,),
                dtype=np.float32)
            
            self.observation_space = gym.spaces.Box(
                low=np.array([0.00, 0.00, self._PHI_RANGE[0], min_phi_end, min_t_gradient]),
                high=np.array([float(max_scouting_runs), np.inf, self._PHI_RANGE[1], self._PHI_RANGE[1], np.inf]), 
                shape=(5,),
                dtype=np.float32)
    
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([self._PHI_RANGE[0], min_phi_end, min_t_gradient]),
                high=np.array([self._PHI_RANGE[1], self._PHI_RANGE[1], max_t_gradient]),
                shape=(3,),
                dtype=np.float32)
            
            self.observation_space = gym.spaces.Box(
                low=np.array([0.00, self._PHI_RANGE[0], min_phi_end, min_t_gradient]),
                high=np.array([np.inf, self._PHI_RANGE[1], self._PHI_RANGE[1], np.inf]), 
                shape=(4,),
                dtype=np.float32)

        self.result = None
        self._history = None
        self._rewards = []
        self._lengths = []
        self._total_episodes = 0
        self._total_steps = 0

        if tf_summary_path is None:
            self._tf_summary_writer = None
        else:
            self._tf_summary_writer = tf.summary.create_file_writer(
                tf_summary_path)
            
    def reset(
        self, 
        compound: tuple[float, float, float] = None, 
        seed: int = None
    ) -> tuple[np.ndarray, dict]:

        super().reset(seed=seed)

        self._total_episodes += 1
        self._episodic_length = 0
        self._episodic_reward = 0.0

        self._history = {
            'retention_factor': [],
            'phi_start': [],
            'phi_end': [],
            'gradient_duration': [],
        }

        self._compound = get_compound() if compound is None else compound

        if self.stop_action:
            initial_state = np.array([0.0, -1.0, -1.0, -1.0, -1.0])
        else:
            initial_state = np.array([-1.0, -1.0, -1.0, -1.0])
        
        return initial_state, {}

    def step(self, actions) -> tuple:
        
        actions = list(actions)

        self._episodic_length += 1

        # Terminate automatically when maximum number of scouting runs is reached
        terminal = self._episodic_length == self.max_scouting_runs

        if self.stop_action:
            # Obtain stop action
            terminate = actions.pop(0)
            if terminate > 0.5 and self._episodic_length < 3:
                 # If agent decides to stop prematurely, force the agent
                 # to continue anyways.
                 terminate = np.random.uniform(0.0, 0.5)

            terminal = terminal or terminate > 0.5

        # Obtain remaining actions
        phi_start, phi_end_, t_gradient = list(actions)

        if self.enforce_constraints:
            phi_incr = phi_end_
            phi_end = np.clip(phi_start + phi_incr, *self._PHI_RANGE)
        else:
            phi_end = phi_end_

        # Reverse the log(t_{G} + 1): exp(t'_{G}) - 1
        t_gradient = np.expm1(t_gradient)

        # For debugging:
        if self._tf_summary_writer is not None:
            with self._tf_summary_writer.as_default(self._total_steps):
                tf.summary.scalar('phi_start', phi_start)
                tf.summary.scalar('phi_end', phi_end)
                tf.summary.scalar('t_gradient', t_gradient)
                self._total_steps += 1

        # Run the simulation (retention models)
        retention_factor = utils.nk_gradient_model(
            self._compound, phi_start, phi_end, t_gradient)[0]
        retention_factor = apply_noise(
            retention_factor, sigma=self._RT_NOISE)

        self._history['retention_factor'].append(retention_factor)
        self._history['phi_start'].append(phi_start)
        self._history['phi_end'].append(phi_end)
        self._history['gradient_duration'].append(t_gradient)

        reward = self._compute_reward() if terminal else 0.0

        # Update state
        t_gradient = np.log1p(t_gradient)
        # Also log retention factor as the magnitude is very high
        retention_factor = np.log1p(retention_factor)
        state = [retention_factor, phi_start, phi_end_, t_gradient]

        if self.stop_action:
            state.insert(0, float(self._episodic_length))

        state = np.array(state, np.float32)

        self._episodic_reward += reward

        if self.stop_action:
            actions.insert(0, terminate)

        if terminal:
            
            self._rewards.append(self._episodic_reward)
            self._lengths.append(self._episodic_length)

            if len(self._rewards) > 100:
                self._rewards.pop(0)
                self._lengths.pop(0)

            if self._tf_summary_writer is not None:
                with self._tf_summary_writer.as_default(self._total_episodes):
                    tf.summary.scalar(
                        'Episode reward', self._episodic_reward)
                    tf.summary.scalar(
                        'Episode length', self._episodic_length)
                    tf.summary.scalar(
                        'Epsiode error', self.result.get('error_param', 0.0))
            
        truncated = terminal
        info = {'action': actions}

        return state, reward, terminal, truncated, info

    def _compute_reward(self) -> float:

        self.result = utils.fit_evaluate_retention_model(
            y_true=self._history['retention_factor'],
            phi_start=self._history['phi_start'],
            phi_end=self._history['phi_end'],
            t_gradient=self._history['gradient_duration'],
            true_param=self._compound, # target parameters
            dwell_time=self._DWELL_TIME,
            void_time=self._VOID_TIME,
        )

        self.result['reward'] = np.clip(
            0.2 / self.result["error_param"], 0.0, 100.0)

        retention_times = (
            0.258 * np.array(self._history['retention_factor']) + 0.258)

        runtimes = [
            max(a, b + 0.614 + 0.258) for (a, b) 
            in zip(retention_times, self._history['gradient_duration'])
        ]
        penalty = sum(runtimes) * self.penalty

        if self.enforce_constraints:
            gradient_ends = (
                np.array(self._history['gradient_duration']) + 0.614 + 0.258)
            penalty += sum(np.where(retention_times > gradient_ends, 100.0, 0.0))

        self.result['penalty'] = penalty

        return self.result['reward'] - self.result['penalty']

    @property
    def pbar_description(self) -> str:
        return (
            f'Average reward: {np.mean(self._rewards):.2f} - '
            f'Average length: {np.mean(self._lengths):.2f}'
        )


class SingleStepScoutingRuns(gym.Env):

    _PHI_RANGE = (0.05, 0.95)
    _RT_NOISE = 0.002 # Inherent system noise obtained from real experiments, may vary
    _VOID_TIME = 0.258 # values obtained from real experiments, may vary
    _DWELL_TIME = 0.614 # values obtained from real experiments, may vary
    
    def __init__(
        self,
        num_scouting_runs: int = 3,
        penalty: float = 0.1,
        enforce_constraints: bool = False,
        tf_summary_path: str = None
    ):
        super().__init__()

        self.num_scouting_runs = num_scouting_runs
        self.enforce_constraints = enforce_constraints
        self.penalty = penalty
        
        min_t_gradient = 1.0 if self.enforce_constraints else 0.1
        max_t_gradient = 6.4 
        min_phi_end = 0.00 if self.enforce_constraints else 0.05 

        n = self.num_scouting_runs 

        self.action_space = gym.spaces.Box(
            low=np.array(
                [self._PHI_RANGE[0]] * n + [min_phi_end] * n + [min_t_gradient] * n
            ),
            high=np.array(
                [self._PHI_RANGE[1]] * n + [self._PHI_RANGE[1]] * n + [max_t_gradient] * n
            ),
            shape=(3 * n, ),
            dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([np.inf, np.inf, np.inf]),
            shape=(3,),
            dtype=np.float32)

        self.result = None
        self._history = None
        self._rewards = []
        self._lengths = []
        self._total_episodes = 0
        self._total_steps = 0

        if tf_summary_path is None:
            self._tf_summary_writer = None
        else:
            self._tf_summary_writer = tf.summary.create_file_writer(
                tf_summary_path)

    def reset(self, compound=None, seed=None) -> tuple:
        
        super().reset(seed=seed)

        self._total_episodes += 1
        self._episodic_length = 0
        self._episodic_reward = 0.0

        self._compound = get_compound() if compound is None else compound

        self._history = {
            'retention_factor': [],
            'phi_start': [],
            'phi_end': [],
            'gradient_duration': [],
        }

        compound = self._compound.copy()

        # Normalize to approximately a 0 and 1 range
        compound[2] = np.log10(compound[2]) / 6.0  # kw
        compound[0] = compound[0] / 50.0           # S1
        compound[1] = compound[1] / 2.5            # S2
        self._state = np.array(compound, np.float32)
        return self._state, {}

    def step(self, actions) -> tuple:
        
        self._episodic_length += 1

        terminal = truncated = True

        n = self.num_scouting_runs

        phi_start = actions[:n]
        phi_end = actions[n:n+n]

        if self.enforce_constraints:
            phi_incr = phi_end
            phi_end = np.clip(phi_start + phi_incr, *self._PHI_RANGE)

        t_gradient = np.expm1(actions[n+n:]) # reverse log

        for ps, pe, tg in zip(phi_start, phi_end, t_gradient):
            k = utils.nk_gradient_model(self._compound, ps, pe, tg)[0]
            k = apply_noise(k, sigma=self._RT_NOISE)

            self._history['retention_factor'].append(k)
            self._history['phi_start'].append(ps)
            self._history['phi_end'].append(pe)
            self._history['gradient_duration'].append(tg)

            # For debugging:
            if self._tf_summary_writer is not None:
                with self._tf_summary_writer.as_default(self._total_steps):
                    tf.summary.scalar('phi_start', ps)
                    tf.summary.scalar('phi_end', pe)
                    tf.summary.scalar('t_gradient', tg)
                    self._total_steps += 1
            
        reward = self._compute_reward()

        self._episodic_reward += reward
        
        if terminal:
            self._rewards.append(self._episodic_reward)
            self._lengths.append(self._episodic_length)

            if len(self._rewards) > 100:
                self._rewards.pop(0)
                self._lengths.pop(0)

            if self._tf_summary_writer is not None:
                with self._tf_summary_writer.as_default(self._total_episodes):
                    tf.summary.scalar(
                        'Episode reward', self._episodic_reward)
                    tf.summary.scalar(
                        'Episode length', self._episodic_length)
                    tf.summary.scalar(
                        'Epsiode error', self.result.get('error_param', 0.0))

                        
        return self._state, reward, terminal, truncated, {}

    def _compute_reward(self) -> float:

        self.result = utils.fit_evaluate_retention_model(
            y_true=self._history['retention_factor'],
            phi_start=self._history['phi_start'],
            phi_end=self._history['phi_end'],
            t_gradient=self._history['gradient_duration'],
            true_param=self._compound, # target parameters
            dwell_time=self._DWELL_TIME,
            void_time=self._VOID_TIME,
        )

        self.result['reward'] = np.clip(
            0.2 / self.result["error_param"], 0.0, 100.0)

        retention_times = (
            0.258 * np.array(self._history['retention_factor']) + 0.258)

        runtimes = [
            max(a, b + 0.614 + 0.258) for (a, b) 
            in zip(retention_times, self._history['gradient_duration'])
        ]
        penalty = sum(runtimes) * self.penalty

        if self.enforce_constraints:
            gradient_ends = (
                np.array(self._history['gradient_duration']) + 0.614 + 0.258)
            penalty += sum(np.where(retention_times > gradient_ends, 100.0, 0.0))

        self.result['penalty'] = penalty
        
        return self.result['reward'] - self.result['penalty']

    @property
    def pbar_description(self) -> str:
        return (
            f'Average reward: {np.mean(self._rewards):.2f} - '
            f'Average length: {np.mean(self._lengths):.2f}'
        )


def get_compound():
    get_random_S1 = lambda: (
        np.random.uniform(10.0, 50.0)
    )
    get_random_S2 = lambda S1: (
        (np.log10(S1) * 2.501 - 2.082) + np.random.uniform(-0.35, 0.35)
    )
    get_random_k0 = lambda S1: (
        10 ** ((S1 * 0.08391 + 0.50544) + np.random.uniform(-1.30, 1.30))
    )

    S1 = get_random_S1()
    S2 = get_random_S2(S1)
    k0 = get_random_k0(S1)

    return np.array([S1, S2, k0], dtype=np.float32)

def apply_noise(retention_factor, mu=1.0, sigma=0.002):
    return retention_factor * (
        mu + np.random.randn(*retention_factor.shape) * sigma)

        