import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import numpy as np

import argparse

from src.agent import DDPGAgent
from src.envs import ScoutingRuns
from src.envs import SingleStepScoutingRuns


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_scouting_runs', type=int, default=3)
    parser.add_argument('--penalty', type=float, default=0.1)
    parser.add_argument('--enforce_constraints', type=int, default=0)
    parser.add_argument('--stop_action', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', type=str, default='ddpg_agent')

    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    tf_summary_path = f'../logs/{args.name}'
    model_save_path = f'../outputs/{args.name}' 

    env = ScoutingRuns(
        max_scouting_runs=int(args.max_scouting_runs),
        penalty=float(args.penalty),
        enforce_constraints=bool(args.enforce_constraints),
        stop_action=bool(args.stop_action),
        tf_summary_path=tf_summary_path)

    # env = SingleStepScoutingRuns(
    #     num_scouting_runs=int(args.max_scouting_runs),
    #     penalty=float(args.penalty),
    #     enforce_constraints=bool(args.enforce_constraints),
    #     tf_summary_path=tf_summary_path)
    
    agent = DDPGAgent(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_bounds=list(zip(env.action_space.low, env.action_space.high)),
        twin_delayed=True,
        units=256,
        activation='relu',
        dropout=None,
        batch_norm=False, 
        batch_size=128,
        initial_learning_rate=1e-4,
        learning_rate_decay_steps=10_000,
        end_learning_rate=1e-5,
        update_after=512,
        buffer_size=10_000,
        gamma=0.99,
        tau=0.005,
        initial_actor_noise=0.2,
        noise_decay_steps=10_000,
        end_actor_noise=0.1,
        target_actor_noise=0.2,
        policy_delay=1,
        start_steps=1_000,
        save_path=model_save_path)

    agent.train(env, num_episodes=10_000)
