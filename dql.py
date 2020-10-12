"""

This module trains a neural network to play mancala using deep Q learning.
The algorithm is mostly just set up here via method calls to external libraries.
The games are played and rewards are calculated in Env.py.

The weights of the neural network are saved to the dql-networks folder every 10,000 steps.

"""

import gym
from Env import MancalaEnv

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import Input
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

env = MancalaEnv(use_random_opponent=True, testing=True)

model = Sequential()
model.add(Flatten(input_shape=(1, 291)))
model.add(Dense(300, activation="relu"))
model.add(Dense(300, activation="relu"))
model.add(Dense(300, activation="relu"))
model.add(Dense(7, activation="relu"))

print(model.input_shape)

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps", value_max=1, value_min=0.1,
                              value_test=0, nb_steps=50000)

memory = SequentialMemory(limit=1000000, window_length=1)

dqn = DQNAgent(model=model, nb_actions=7, policy=policy, memory=memory,
               gamma=0.99, train_interval=1, delta_clip=1.)

dqn.compile(Adam(lr=0.001), metrics=["mae"])

callbacks = [ModelIntervalCheckpoint("dql-networks/step-{step}.h5f", interval=10000)]

dqn.fit(env, callbacks=callbacks, nb_steps=1000000, log_interval=10000, nb_max_start_steps=0)

dqn.save_weights("dql-networks/final.h5f", overwrite=True)