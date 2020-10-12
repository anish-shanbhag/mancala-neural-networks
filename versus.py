"""

  This module tests the neural networks trained by neuroevolution and deep Q learning by making them
play mancala games against each other.

  For every saved generation of the population of neuroevolution-trained networks, the weights of the deep
Q learning network from around the same time after training are extracted and paired up with that generation.

  Then, the population of neuroevolution-trained networks play mancala games against the deep Q
learning network, and the average/best win rate across the population are saved to the results folder:
  - versus-average-winrate.txt
  - versus-best-winrate.txt

"""

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import Input
from keras.optimizers import Adam

import multiprocessing
from joblib import Parallel, delayed

import numpy as np
import pickle

from Env import MancalaEnv

from copy import deepcopy

import random

GAMES = 1000

class NeuralNetwork:
  def __init__ (self, weights, biases):
    self.weights = weights
    self.biases = biases
  def feedforward(self, activations):
    for weight_matrix, bias_vector in zip(self.weights, self.biases):
      activations = self.sigmoid(np.dot(weight_matrix, activations) + bias_vector)
    return activations
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

indices = []

with open("results/neuroevolution-time.txt", "r") as f:
  for line in f:
    time = float(line.strip("\n"))
    latest = 0
    with open("results/dql-time.txt", "r") as f2:
      for i, line2 in enumerate(f2):
        if float(line2) <= time:
          latest = i
        else:
          break
    indices.append(latest)

print(indices)

open("results/versus-average-winrate.txt", "w").close()
open("results/versus-best-winrate.txt", "w").close()

def test_network(network, step):
  model = Sequential()
  model.add(Flatten(input_shape=(1, 291)))
  model.add(Dense(300, activation="relu"))
  model.add(Dense(300, activation="relu"))
  model.add(Dense(300, activation="relu"))
  model.add(Dense(7, activation="relu"))
  model.load_weights(f"dql-networks/step-{step * 10000 + 10000}.h5f")
  env = MancalaEnv(use_random_opponent=False, testing=True)
  network_won = 0
  for _ in range(GAMES):
    # Obtain the initial observation by resetting the environment.
    observation = deepcopy(env.reset())
    done = False
    while not done:
      if random.random() > 0.1:
        if env.current_player:
          action_values = network.feedforward(observation)
        else:
          action_values = model.predict(np.array([[observation]]))[0]
        action = env.get_best_action(action_values)
      else:
        action = env.get_random_action()
      
      observation, r, done, info = env.step(action)
      observation = deepcopy(observation)
    if env.won:
      network_won += 1
  return network_won

for neuroevolution, reinforcement in enumerate(indices):
  if reinforcement * 10000 + 10000 > 620000:
    break

  with open("neuroevolution-networks/generation-" + str(neuroevolution * 10), "rb") as f:
    population = pickle.load(f)
  
  # 'won' refers to how many neuroevolution won

  results = Parallel(n_jobs=12)(delayed(test_network)(network, reinforcement) for network in population)

  average_winrate = sum(results) / len(population) / GAMES
  best_winrate = max(results) / GAMES
  with open("results/versus-average-winrate.txt", "a") as f:
    f.write(str(average_winrate) + "\n")
  with open("results/versus-best-winrate.txt", "a") as f:
    f.write(str(best_winrate) + "\n")
  print(f"Generation {neuroevolution * 10} done")