"""

This module tests the neural network trained using deep Q learning.
It calculates the network's win rate against a random player after every 5,000 steps.

This win rate is output to the results folder (dql-winrate.txt).

"""


from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import Input
from keras.optimizers import Adam

import numpy as np
import glob

from Env import MancalaEnv

from copy import deepcopy

GAMES = 1000

model = Sequential()
model.add(Flatten(input_shape=(1, 291)))
model.add(Dense(300, activation="relu"))
model.add(Dense(300, activation="relu"))
model.add(Dense(300, activation="relu"))
model.add(Dense(7, activation="relu"))

env = MancalaEnv(use_random_opponent=True, testing=True)

open("results/dql-winrate.txt", "w").close()

for f in glob.glob("dql-networks/*.h5f"):
  model.load_weights(f)
  won = 0
  for episode in range(GAMES):
    # Obtain the initial observation by resetting the environment.
    observation = deepcopy(env.reset())

    done = False
    while not done:
      action_values = model.predict(np.array([[observation]]))[0]
      action = env.get_best_action(action_values)
      observation, r, done, info = env.step(action)
      observation = deepcopy(observation)
    if env.won:
      won += 1
  with open("results/dql-winrate.txt", "a") as f:
    f.write(str(won / GAMES) + "\n")