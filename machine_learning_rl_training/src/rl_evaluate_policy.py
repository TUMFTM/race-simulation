"""
author:
André Thomaser

refactored by:
Alexander Heilmeier

date:
01.11.2020

.. description::
The functions in this file are related to the reinforcement learning (RL) approach taken to train a Virtual Strategy
Engineer (VSE) within the race simulation to make reasonable strategy decisions. The main script is located in
main_train_rl_agent_dqn.py.
"""

from machine_learning_rl_training.src.rl_environment_single_agent import RaceSimulation
import helper_funcs.src.progressbar
import numpy as np
import tensorflow as tf
import tf_agents.trajectories


def convert_time_step(time_step) -> tf_agents.trajectories.time_step.TimeStep:
    """
    This function converts a given time_step from a Python environment to a TF time_step.
    """

    time_step = tf_agents.trajectories.time_step.\
        TimeStep(step_type=tf.constant([time_step.step_type], dtype=np.int32),
                 reward=tf.constant([time_step.reward], dtype=np.float32),
                 discount=tf.constant([time_step.discount], dtype=np.float32),
                 observation=tf.convert_to_tensor([time_step.observation], dtype=np.float32))

    return time_step


def print_returns_positions(py_env: RaceSimulation,
                            num_races: int,
                            tf_lite_path: str,
                            vse_others: str = None):
    """
    This function calculates the returns and final positions with a given q_network as tf-lite over num_races (episodes)
    and distinguishes the returns and positions of races with FCY phases and races without FCY phases.

    vse_others: set VSE type of other drivers -> "supervised" or "reinforcement" or "basestrategy" or "realstrategy"
    """

    if vse_others is not None:
        py_env.vse_type = vse_others
    else:
        vse_others = py_env.vse_type

    # initialize TF lite q-network
    q_net_lite = {"interpreter": tf.lite.Interpreter(model_path=tf_lite_path)}
    q_net_lite["interpreter"].allocate_tensors()
    q_net_lite["input_index"] = q_net_lite["interpreter"].get_input_details()[0]['index']
    q_net_lite["output_index"] = q_net_lite["interpreter"].get_output_details()[0]['index']

    fcy_returns = []
    nofcy_returns = []
    fcy_final_positions = []
    nofcy_final_positions = []

    print('INFO: Evaluating reinforcement VSE by average returns and positions over %i races against %s VSE...'
          % (num_races, vse_others))

    for i in range(num_races):
        time_step = py_env.reset()
        episode_return = 0.0

        while not time_step.is_last():
            # set NN input
            q_net_lite["interpreter"].set_tensor(q_net_lite["input_index"], convert_time_step(time_step).observation)

            # invoke NN
            q_net_lite["interpreter"].invoke()

            # fetch NN output
            action_q = q_net_lite["interpreter"].get_tensor(q_net_lite["output_index"])[0].argmax()

            time_step = py_env.step(action_q)
            episode_return += time_step.reward

        final_position = py_env.race.positions[py_env.race.get_last_compl_lap(py_env.idx_driver), py_env.idx_driver]

        if not py_env.race.fcy_data['phases']:
            nofcy_returns.append(episode_return)
            nofcy_final_positions.append(final_position)
        else:
            fcy_returns.append(episode_return)
            fcy_final_positions.append(final_position)

        helper_funcs.src.progressbar.progressbar(i=i + 1, i_total=num_races, prefix='INFO: Progress:')

    print('RESULT: Average return (total): %.3f (FCY: %.3f, no FCY: %.3f),'
          ' average position (total): %.1f (FCY: %.1f, no FCY: %.1f),'
          ' FCY races: %i, no FCY races: %i'
          % (float(np.mean(fcy_returns + nofcy_returns)),
             float(np.mean(fcy_returns)),
             float(np.mean(nofcy_returns)),
             float(np.mean(fcy_final_positions + nofcy_final_positions)),
             float(np.mean(fcy_final_positions)),
             float(np.mean(nofcy_final_positions)),
             len(fcy_returns),
             len(nofcy_returns)))
