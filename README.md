# Twin-delayed DDPG for optimal selection of gradient scouting runs

## About
An attempt to implement and train a DDPG agent to learn to select optimal scouting runs for a given compound. The scouting runs are run in a simulator, using well-studied retention models. 

> Caution: the agent is not meant to be used in real practice. 

The goal of this project was two-fold:

1. To succesfully develop and train a reinforcement learning (RL) agent to perform scouting runs based on feedback. The feedback is computed based on a reward function which takes into consideration the accuracy of the retention models (fit to/resulted from the scouting runs) and the run-time.
2. If the agent learns well, get insight on what scouting runs are optimal given a certain compound. Are the choices of the agent what we expect? Are there any surprises?

## Room for improvement
Although occasionally converging to reasonable solutions, the training is somwhat unreliable and unstable. One of the main reasons for this is likely way the rewards are calculated; namely, that it is based on retention model fitting, which is highly stochastic and varies significantly depending on what scouting runs were made. (This is also an issue when later evaluating the performance.) Below are some suggestions for improving the DDPG algorithm:

1. Modify the reward function, including better scaling (e.g., between -1.0 and 1.0)
2. Scale actions between -1.0 and 1.0, and scale states between e.g. 0.0 and 1.0. Caution: need to reverse scaling in the environment.
3. Fine-tune the hyperparameters of the DDPG agent
    - E.g. discount factor,
    - learning rate,
    - action noise,
    - and tau.
4. Improve the architecture of the neural networks, as well as its hyperparameters
    - E.g. better initialization,
    - regularizaton,
    - number of layers and units.
5. Replace existing buffer with prioritized experience replay buffer.

## Requirements
* Python 3.10
    * jupyter (version 1.0.0)
    * tensorflow (version 2.13.0)
    * matplotlib (version 3.7.2)
    * tqdm (version 4.45.0)
    * gymnasium (version 0.26.2)
    
> See setup.py for more detail on what packages are installed and what versions.

## Setup and run
1. Navigate to to the desired location.
2. Clone the repository: e.g., `git clone git@github.com:pharmanlysis/ddpg-optimal-scouting-runs.git`
3. Install the package (setup the repistory): `pip install -e .`
4. Navigate source code (in `src/`) to study and possibly modify the code. 
5. Navigate to scripts (`../scripts/`) and train the agent in the environment, via `python main.py`
6. Navigate to root (`../`) and track the training progression via tensorboard: `tensorboard --logdir logs/`
