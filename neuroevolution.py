"""

  This module trains a population of neural networks to play mancala using neuroevolution.

  The training time and average/best win rate against a random player for each generation are output to the results folder:
  - neuroevolution-average-winrate.txt
  - neuroevolution-best-winrate.txt
  - neuroevolution-time.txt

  The weights of the neural networks are also saved in the neuroevolution-networks folder after each generation.

"""

import numpy as np
from numpy import random
from itertools import chain
import json
import multiprocessing
import time
import pickle
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

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

class GameResult:
  def __init__(self, winner, scores):
    self.winner = winner
    self.scores = scores

class RandomPlayer:
  def __init__ (self):
    pass
  def feedforward(self, activations):
    return random.rand(7)

population_size = 200

random_player = RandomPlayer()

def genetic_algorithm(layer_size, layer_size_2, layer_size_3,
                      weight_init_stdev, weight_mutation_chance, weight_mutation_stdev,
                      bias_init_stdev, bias_mutation_chance, bias_mutation_stdev):
  def simulate():
    population = []
    for _ in range(0, population_size):
      weights = [np.random.normal(0, weight_init_stdev, (layer_size, 291))]
      if layer_size_2 != 0:
        weights.append(np.random.normal(0, weight_init_stdev, (layer_size_2, layer_size)))
      if layer_size_3 != 0:
        weights.append(np.random.normal(0, weight_init_stdev, (layer_size_3, layer_size_2)))
        weights.append(np.random.normal(0, weight_init_stdev, (7, layer_size_3)))
      elif layer_size_2 != 0:
        weights.append(np.random.normal(0, weight_init_stdev, (7, layer_size_2)))
      else:
        weights.append(np.random.normal(0, weight_init_stdev, (7, layer_size)))
      
      # rows of matrix are hidden/outputs, while columns are inputs

      biases = [np.random.normal(0, bias_init_stdev, layer_size)]
      if layer_size_2 != 0:
        biases.append(np.random.normal(0, bias_init_stdev, layer_size_2))
      if layer_size_3 != 0:
        biases.append(np.random.normal(0, bias_init_stdev, layer_size_3))
      
      biases.append(np.random.normal(0, bias_init_stdev, 7))

      population.append(NeuralNetwork(weights, biases))

    def play_game(player1, player2):
      # board structure: start counting at leftmost pocket on player's side
      # go counterclockwise, and end at opponent's mancala
      board = np.array([4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0])
      currentPlayer = player1 if random.choice([True, False]) else player2

      turn = 0
      pie_rule_available = False

      while not np.all(board[0:6] == 0) and not np.all(board[7:13] == 0):
        currentPlayer = player1 if currentPlayer == player2 else player2
        # flip board orientation every turn
        board = np.concatenate([board[7:14], board[0:7]])
        turn += 1
        while True:
          observation = [0] * 288
          for i, pieces in enumerate(list(board[0:6]) + list(board[7:13])):
            observation[pieces + (i * 24)] = 1
          observation += [board[6], board[13], 1 if turn == 1 else 0]

          outputs = currentPlayer.feedforward(observation)
          move = 0
          move_confidence = -1
          for i in range(0, 7):
            if outputs[i] > move_confidence:
              if i == 6:
                if turn == 2 and pie_rule_available:
                  move = i
                  move_confidence = outputs[move]
              elif board[i] != 0:
                move = i
                move_confidence = outputs[move]

          def resolve_move(board, pie_rule_available):
            if move == 6:
              board = np.concatenate([board[7:14], board[0:7]])
              return False
            if turn == 2:
              pie_rule_available = False
            pocket = move
            while True:
              pieces = board[pocket]
              if pieces == 0:
                # Invalid move, assume game is over because all pockets are empty
                return False
              board[pocket] = 0
              for _ in range(pieces):
                pocket = pocket + 1 if pocket < 12 else 0
                board[pocket] = board[pocket] + 1
              if pocket == 6:
                return True
              if board[pocket] == 1:
                if pocket < 6:
                  board[6] += board[12 - pocket] + 1
                  board[pocket] = 0
                  board[12 - pocket] = 0
                return False
            
          free_move = resolve_move(board, pie_rule_available)
          if free_move:
            continue
          else:
            break
      
      mancala_to_fill = 13 if np.all(board[0:6] == 0) else 6
      board[mancala_to_fill] += np.sum(np.concatenate([board[0:6], board[7:13]]))
      winner_player = currentPlayer if board[6] > board[13] else (player1 if currentPlayer == player2 else player2)
      winner = player1 == winner_player
      scores = [board[6] if currentPlayer == player1 else board[13], board[6] if currentPlayer == player2 else board[13]]
      return GameResult(winner, scores)

    generations = 1000

    # used later to evaluate fitness
    def play_against_random(player, games):
      won = 0
      for _ in range(games):
        game = play_game(player, random_player)
        if game.scores[0] > game.scores[1]:
          won += 1
      return won

    start = time.time()

    open("results/neuroevolution-time.txt", "w").close()
    open("results/neuroevolution-average-winrate.txt", "w").close()
    open("results/neuroevolution-best-winrate.txt", "w").close()
    
    for generation in range(0, generations):
      
      shuffled_population = population.copy()
      random.shuffle(population)
      # the networks play games against a random player
      # the networks that win more games move on to become parents of the next generation
      fitness_games = max(100, generation)

      def select_from_pair(player1, player2):

        score1 = play_against_random(player1, fitness_games)
        score2 = play_against_random(player2, fitness_games)

        return player1 if score1 > score2 else player2, score1, score2

      pairs = []
      for _ in range(population_size // 2):
        pairs.append((shuffled_population.pop(), shuffled_population.pop()))
      results = Parallel(n_jobs=num_cores)(delayed(select_from_pair)(pair[0], pair[1]) for pair in pairs)
      winners, scores1, scores2 = [list(a) for a in zip(*results)]

      if generation % 10 == 0:
        scores = scores1 + scores2

        generation_time = time.time() - start
        average_score = sum(scores) / population_size / fitness_games
        best_score = max(scores) / fitness_games

        with open("results/neuroevolution-time.txt", "a") as f:
          f.write(str(generation_time) + "\n")
        with open("results/neuroevolution-average-winrate.txt", "a") as f:
          f.write(str(average_score) + "\n")
        with open("results/neuroevolution-best-winrate.txt", "a") as f:
          f.write(str(best_score) + "\n")
        with open(f"neuroevolution-networks/generation-{generation}", 'wb+') as f:
          pickle.dump(population, f)

      # breeding (crossover/mutation)
      population = winners.copy()
      for _ in range(2):
        winners_copy = winners.copy()
        random.shuffle(winners_copy)
        pairs = []
        while winners_copy:
          pairs.append((winners_copy.pop(), winners_copy.pop()))
        def crossover_and_mutate(parent1, parent2):
          child_weights = []
          child_biases = []
          for i in range(len(parent1.weights)):
            weights_shape = parent1.weights[i].shape
            weights_crossover = random.choice([True, False], weights_shape)
            child_weights.append(parent1.weights[i].copy())
            child_weights[i][weights_crossover] = parent2.weights[i][weights_crossover]
            weights_randomized = random.normal(0, weight_mutation_stdev, weights_shape)
            weights_mutated = random.choice([True, False], weights_shape,
                                            p=[weight_mutation_chance, 1 - weight_mutation_chance])
            child_weights[i][weights_mutated] += weights_randomized[weights_mutated]

            biases_shape = parent1.biases[i].shape
            biases_crossover = random.choice([True, False], biases_shape)
            child_biases.append(parent1.biases[i].copy())
            child_biases[i][biases_crossover] = parent2.biases[i][biases_crossover]
            biases_randomized = random.normal(0, bias_mutation_stdev, biases_shape)
            biases_mutated = random.choice([True, False], biases_shape,
                                            p=[bias_mutation_chance, 1 - bias_mutation_chance])
            child_biases[i][biases_mutated] += biases_randomized[biases_mutated]
          return NeuralNetwork(child_weights, child_biases)
        children = Parallel(n_jobs=num_cores)(delayed(crossover_and_mutate)(pair[0], pair[1]) for pair in pairs)
        population += children
      print(f"Generation {generation} complete.")
      if False:
        generation_games = 1000
        result = Parallel(n_jobs=num_cores)(delayed(play_against_random)(player, generation_games) for player in population)
        generation_win_percentage = sum(result) / generation_games / population_size
        print(f"Generation {generation}: " + str(generation_win_percentage))
        print("Best performing individual: " + str(max(result) / generation_games))
      
    # below computes total_won after all generations are complete
    trained_games = 1000
    trained_won_games = sum(Parallel(n_jobs=num_cores)(delayed(play_against_random)(player, trained_games) for player in population))
    trained_win_percentage = trained_won_games / trained_games / population_size

    print("After training: " + str(trained_win_percentage))
    return trained_win_percentage
  simulated_win_percentage = simulate()
  return simulated_win_percentage

if __name__ == "__main__":
  # Hyperparameters used for neuroevolution
  layer_1_size = 300
  layer_2_size = 300
  layer_3_size = 300
  weight_init_stdev = 0.1
  weight_mutation_chance = 0.2
  weight_mutation_stdev = 0.1
  bias_init_stdev = 0.1
  bias_mutation_chance = 0.2
  bias_mutation_stdev = 0.1

  genetic_algorithm(layer_1_size, layer_2_size, layer_3_size,
                    weight_init_stdev, weight_mutation_chance, weight_mutation_stdev,
                    bias_init_stdev, bias_mutation_chance, bias_mutation_stdev)