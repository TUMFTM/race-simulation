"""
author:
André Thomaser

refactored by:
Alexander Heilmeier

date:
01.11.2020

.. description::
This is the main script for training a Virtual Strategy Engineer (VSE) within the race simulation to make reasonable
race strategy decisions for the specified race. This is possible by a reinforcement learning approach. Please refer to
the readme (and the linked papers) for further information.

The estimated duration for the training of a single agent against the supervised VSE with the default settings is about
an hour. Currently, the process is not parallelized and, thus, uses only a single CPU core.
"""

import machine_learning_rl_training
import os
import pickle
import shutil
import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import racesim


# ----------------------------------------------------------------------------------------------------------------------
# USER INPUT -----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# environment parameters
race = "Shanghai_2019"  # set race (see racesim/input/parameters for possible races)
# VSE type for other drivers: 'basestrategy', 'realstrategy', 'supervised', 'reinforcement' (if already available),
# 'multi_agent' (if VSE should learn for all drivers at once)
vse_others = "supervised"
mcs_pars_file = "pars_mcs.ini"  # parameter file for Monte Carlo parameters

# hyperparameters
num_iterations = 250_000
replay_buffer_max_length = 200_000
initial_collect_steps = 200
collect_steps_per_iteration = 1

fc_layer_params = (64, 64,)
batch_size = 64
learning_rate = 1e-3
gamma = 1.0  # discount rate
n_step_update = 1
target_update_period = 1
dueling_q_net = False

# training options
log_interval = 100_000
eval_interval = 50_000
num_eval_episodes = 100

# postprocessing (currently not implemented for multi-agent environment)
calculate_final_positions = False  # activate or deactivate evaluation after training
num_races_postproc = 10_000
# VSE type for other drivers: 'basestrategy', 'realstrategy', 'supervised', 'reinforcement' (if already available)
vse_others_postproc = "supervised"


# ----------------------------------------------------------------------------------------------------------------------
# CHECK USER INPUT -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if vse_others == "multi_agent" and calculate_final_positions:
    print("WARNING: Evaluation of trained strategy is currently not implemented for the multi-agent environment!"
          " Setting calculate_final_positions = False!")
    calculate_final_positions = False

# ----------------------------------------------------------------------------------------------------------------------
# CHECK FOR WET RACE ---------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# create race parameter file name
race_pars_file = 'pars_%s.ini' % race

# load parameter file
pars_in = racesim.src.import_pars.import_pars(use_print=False,
                                              use_vse=False,
                                              race_pars_file=race_pars_file,
                                              mcs_pars_file=mcs_pars_file)[0]

# loop through drivers and check for intermediate or wet tire compounds in real race
for driver in pars_in["driver_pars"]:
    if any([True if strat[1] in ["I", "W"] else False for strat in pars_in["driver_pars"][driver]["strategy_info"]]):
        raise RuntimeError("Cannot train for current race %s because it was a (partly) wet race!" % race)

# ----------------------------------------------------------------------------------------------------------------------
# ENVIRONMENT ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if vse_others == 'multi_agent':
    train_py_env = machine_learning_rl_training.src.rl_environment_multi_agent.\
        RaceSimulation(race_pars_file=race_pars_file,
                       mcs_pars_file=mcs_pars_file,
                       use_prob_infl=True,
                       create_rand_events=True)
    eval_py_env = machine_learning_rl_training.src.rl_environment_multi_agent. \
        RaceSimulation(race_pars_file=race_pars_file,
                       mcs_pars_file=mcs_pars_file,
                       use_prob_infl=True,
                       create_rand_events=True)
else:
    train_py_env = machine_learning_rl_training.src.rl_environment_single_agent. \
        RaceSimulation(race_pars_file=race_pars_file,
                       mcs_pars_file=mcs_pars_file,
                       vse_type=vse_others,
                       use_prob_infl=True,
                       create_rand_events=True)
    eval_py_env = machine_learning_rl_training.src.rl_environment_single_agent \
        .RaceSimulation(race_pars_file=race_pars_file,
                        mcs_pars_file=mcs_pars_file,
                        vse_type=vse_others,
                        use_prob_infl=True,
                        create_rand_events=True)

train_tf_env = tf_py_environment.TFPyEnvironment(environment=train_py_env)
eval_tf_env = tf_py_environment.TFPyEnvironment(environment=eval_py_env)

print('INFO: Race: %s, strategy of other drivers: %s' % (race, vse_others))

if train_py_env.batched:
    print('INFO: Batched environment:', train_py_env.batched(), 'batch size:', train_py_env.batch_size)

print('INFO: Observation spec:', train_py_env.time_step_spec().observation)
print('INFO: Action spec:', train_py_env.action_spec())


# ----------------------------------------------------------------------------------------------------------------------
# DQN-AGENT ------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

q_net = q_network.QNetwork(input_tensor_spec=train_tf_env.observation_spec(),
                           action_spec=train_tf_env.action_spec(),
                           fc_layer_params=fc_layer_params)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)
boltzmann_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=1.0,
                                                             decay_steps=num_iterations,
                                                             end_learning_rate=0.01)

agent = dqn_agent.DqnAgent(time_step_spec=train_tf_env.time_step_spec(),
                           action_spec=train_tf_env.action_spec(),
                           q_network=q_net,
                           optimizer=optimizer,
                           n_step_update=n_step_update,
                           target_update_period=target_update_period,
                           td_errors_loss_fn=common.element_wise_squared_loss,
                           gamma=gamma,
                           train_step_counter=train_step_counter,
                           epsilon_greedy=None,
                           boltzmann_temperature=lambda: boltzmann_fn(train_step_counter))
agent.initialize()


# ----------------------------------------------------------------------------------------------------------------------
# POLICIES -------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(time_step_spec=train_tf_env.time_step_spec(),
                                                action_spec=train_tf_env.action_spec())


# ----------------------------------------------------------------------------------------------------------------------
# REPLAY BUFFER --------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                               batch_size=train_tf_env.batch_size,
                                                               max_length=replay_buffer_max_length)


# ----------------------------------------------------------------------------------------------------------------------
# DATA COLLECTION ------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def collect_step(env: tf_py_environment.TFPyEnvironment, policy, buffer):
    time_step_ = env.current_time_step()
    action_step = policy.action(time_step_)
    next_time_step = env.step(action=action_step.action)
    traj = trajectory.from_transition(time_step=time_step_,
                                      action_step=action_step,
                                      next_time_step=next_time_step)

    # add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env: tf_py_environment.TFPyEnvironment, policy, buffer, steps: int):
    for _ in range(steps):
        collect_step(env=env, policy=policy, buffer=buffer)


collect_data(env=train_tf_env, policy=random_policy, buffer=replay_buffer, steps=initial_collect_steps)

# debugging info if required
# iter(replay_buffer.as_dataset()).next()

dataset = replay_buffer.as_dataset(num_parallel_calls=3,
                                   sample_batch_size=batch_size,
                                   num_steps=n_step_update + 1).prefetch(3)

dataset_iterator = iter(dataset)

# debugging info if required
# dataset_iterator.next()


# ----------------------------------------------------------------------------------------------------------------------
# METRICS AND EVALUATION -----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def compute_average_return(env: tf_py_environment.TFPyEnvironment,
                           policy,
                           num_episodes: int = 1) -> float:
    total_return = 0.0

    for _ in range(num_episodes):
        time_step_ = env.reset()
        episode_return = 0.0

        while not any(time_step_.is_last()):
            action_step = policy.action(time_step_)
            time_step_ = env.step(action=action_step.action)
            episode_return += np.mean(time_step_.reward)

        total_return += episode_return

    average_return = total_return / num_episodes

    return average_return


# ----------------------------------------------------------------------------------------------------------------------
# TRAINING THE AGENT ---------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

print("INFO: Starting training...")

# (optional) optimize by wrapping some of the code in a graph using TF function
agent.train = common.function(agent.train)

# reset the train step
agent.train_step_counter.assign(0)

# evaluate the agent's policy once before training
avg_return = compute_average_return(env=eval_tf_env, policy=eval_policy, num_episodes=num_eval_episodes)
print("INFO: Evaluated the agent's policy once before the training, average return: %.3f" % avg_return)

for _ in range(num_iterations):
    # collect a few steps using collect_policy and save them to the replay buffer
    for _ in range(collect_steps_per_iteration):
        collect_step(env=train_tf_env, policy=collect_policy, buffer=replay_buffer)

    # sample a batch of data from the buffer and update the agent's network
    experience = next(dataset_iterator)[0]
    train_loss = agent.train(experience=experience).loss

    step = int(agent.train_step_counter.numpy())

    if step % log_interval == 0:
        print('INFO: Step %i, loss %.3f' % (step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_average_return(env=eval_tf_env, policy=eval_policy, num_episodes=num_eval_episodes)
        print('INFO: Step %i, average return %.3f' % (step, avg_return))

    if (10 * step) % num_iterations == 0:
        # print every 10%
        print('INFO: Training progress: %.0f%%...' % (step / num_iterations * 100.0))


# ----------------------------------------------------------------------------------------------------------------------
# SAVE Q-NETWORK AS TF-LITE --------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# set paths
repo_path_ = os.path.dirname(os.path.abspath(__file__))
output_path_ = os.path.join(repo_path_, 'machine_learning_rl_training', 'output')
tmp_path_ = os.path.join(output_path_, 'tmp')

# create folders if not existing
os.makedirs(output_path_, exist_ok=True)
os.makedirs(tmp_path_, exist_ok=True)

# set file paths
tmp_file_path_ = os.path.join(tmp_path_, 'q_network_tmp')
preprocessor_file_path_ = os.path.join(output_path_, "preprocessor_reinforcement_%s.pkl" % race)
qnet_file_path_ = os.path.join(output_path_, "nn_reinforcement_%s.tflite" % race)

# save preprocessor
with open(preprocessor_file_path_, "wb") as fh:
    pickle.dump(train_py_env.cat_preprocessor, fh)

# DQN must be save and loaded once such that the conversion to TF lite works
agent._target_q_network.save_weights(tmp_file_path_)
q_net = q_network.QNetwork(input_tensor_spec=train_tf_env.observation_spec(),
                           action_spec=train_tf_env.action_spec(),
                           fc_layer_params=fc_layer_params)
q_net.load_weights(tmp_file_path_)
time_step = eval_tf_env.reset()
q_net(time_step.observation)
shutil.rmtree(tmp_path_)  # delete tmp_path_ again

# convert to TF lite model
converter = tf.lite.TFLiteConverter.from_keras_model(q_net)
tflite_q_net = converter.convert()

# save TF lite model
with open(qnet_file_path_, 'wb') as fh:
    fh.write(tflite_q_net)

print('RESULT: TF-lite q-network and preprocessor were saved in the machine_learning_rl_training/output folder!')

# ----------------------------------------------------------------------------------------------------------------------
# CALCULATE AVERAGE RETURNS --------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if calculate_final_positions:
    py_env = machine_learning_rl_training.src.rl_environment_single_agent.RaceSimulation(race_pars_file=race_pars_file,
                                                                                         mcs_pars_file=mcs_pars_file,
                                                                                         vse_type=vse_others,
                                                                                         use_prob_infl=True,
                                                                                         create_rand_events=True)

    machine_learning_rl_training.src.rl_evaluate_policy.print_returns_positions(py_env=py_env,
                                                                                num_races=num_races_postproc,
                                                                                tf_lite_path=qnet_file_path_,
                                                                                vse_others=vse_others_postproc)
