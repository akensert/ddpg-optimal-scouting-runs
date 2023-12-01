import numpy as np
import matplotlib.pyplot as plt


def run_env(env, policy, initial_state):

    state = initial_state

    actions = []

    while True:
            
        action = policy(state).numpy()

        next_state, reward, done, _, info = env.step(action)

        if env.__class__.__name__ == 'ScoutingRuns':
            action[-1] = np.expm1(action[-1])
            action[1] = env._history['phi_end'][-1]
            actions.append(action)
        else:
            action[2 * env.num_scouting_runs:] = np.expm1(
                action[2 * env.num_scouting_runs:])
            action[env.num_scouting_runs: 2 * env.num_scouting_runs] = env._history['phi_end']
            
            action = [
                action[:env.num_scouting_runs],
                action[env.num_scouting_runs:-env.num_scouting_runs],
                action[-env.num_scouting_runs:]
            ]

            actions = list(zip(*action))
            
        state = next_state.copy()

        if done:
            break

    env.result['actions'] = actions

    return env

def print_results(env):

    if env.__class__.__name__ == 'ScoutingRunsV1':
        env.result['actions'] = list(zip(*np.split(env.result['actions'][0], 3)))

    print("\t   phi_start  phi_end  t_gradient     tR     phi_elution")
    print('---' * 20)
    tr = ((0.258 * env.result['y_true_gra']) + 0.258).round(4)
    pae = env.result['y_true_elution'].round(4)
    for i, (action, t, p) in enumerate(zip(env.result['actions'], tr, pae)):
        print("action {}:  {:.3f}      {:.3f}     {:6.3f}      {:7.2f}   {:.3f}".format(
            i+1, *action, t, p))

    print('\n')
    print("\t      S1       S2      kw")
    print('---' * 20)

    pp, tp = env.result['pred_param'], env.result['true_param']

    print('est. params:  {:.3f}   {:.3f}   {:.0f}'.format(*pp))
    print('true params:  {:.3f}   {:.3f}   {:.0f}'.format(*tp))
    print('\n')

def plot_results(env):

    plt.scatter(
        np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        np.log10(env.result['y_true_iso']), s=50
    )

    plt.plot(
        np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        np.log10(env.result['y_pred_iso']), linewidth=2
    )

    plt.xlabel('$\phi$', fontsize=20)
    plt.ylabel('$k$', fontsize=20)
    plt.title(
        "Performance of resulting retention\nmodel on isocratic data",
        fontsize=16)

