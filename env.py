"""

This module defines an OpenAI Gym environment for mancala.
The board state is stored here and moves are carried in the 'step' method.
Rewards are also calculated for use with the deep Q learning algorithm.

Time taken for each set of 5,000 steps are output to the results folder (dql-time.txt).

"""

import gym
from gym import spaces
import numpy as np
import random
import time

VISUALIZE_MOVES = False

class MancalaEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, use_random_opponent, testing):
    super(MancalaEnv, self).__init__()
    self.start = time.time()
    self.current_step = 0
    self.games_won = 0
    self.games_tied = 0
    self.games = 0
    self.reward_range = (-10000, 100)
    self.action_space = spaces.Discrete(7)
    self.use_random_opponent = use_random_opponent
    if self.use_random_opponent:
      open("results/dql-time.txt", "w").close()
    self.testing = testing
    self.observation_space = spaces.Box(low=0, high=50, shape=(291,), dtype=np.uint8)

  def step(self, move):
    # Execute one time step within the environment

    if self.current_step % 5000 == 0 and self.use_random_opponent:
      with open("results/dql-time.txt", "a") as f:
        f.write(str(time.time() - self.start) + "\n")
    
    self.current_step += 1
    reward = 0

    while True:
      if self.current_player:
        reward = self.move(move)
      elif self.use_random_opponent:
        self.move(self.get_random_action())
      else:
        self.move(move)

      # if one side is completely empty, then the game is over - transfer remaining pieces to the corresponding player's mancala
      if np.all(self.board[0:6] == 0) or np.all(self.board[7:13] == 0):
        mancala_to_fill = 13 if np.all(self.board[0:6] == 0) else 6
        self.board[mancala_to_fill] += np.sum(np.concatenate([self.board[0:6], self.board[7:13]]))
        self.done = True
        
        player_mancala = 6 if self.current_player else 13
        opponent_mancala = 13 if self.current_player else 6
        
        reward = (self.board[player_mancala] - self.board[opponent_mancala])

        if VISUALIZE_MOVES:
          print("\nPlayer total: " + str(self.board[player_mancala]))
          print("Opponent total: " + str(self.board[opponent_mancala]))

        self.won = self.board[player_mancala] > self.board[opponent_mancala]
        
        if VISUALIZE_MOVES:
          self.games += 1
          if self.board[player_mancala] > self.board[opponent_mancala]:
            self.games_won += 1
          elif self.board[player_mancala] == self.board[opponent_mancala]:
            self.games_tied += 1
          if self.games % 1000 == 0:
            print("\nWin percentage:", self.games_won / self.games)
            print("Tie percentage:", self.games_tied / self.games)
            print(self.games)
            self.games = 0
            self.games_won = 0
            self.games_tied = 0

      if self.current_player or self.done or not self.use_random_opponent:
        return self.get_observation(), reward, self.done, {}

  def move(self, move):
    if move == 6:
      if self.turn == 2 and self.pie_rule_available:
        if not self.current_player:
          self.used_pie_rule = True
        # pie rule: switch sides
        self.flip_board()
        self.switch_turn()
        return self.board[13] - self.board[6]
    elif self.turn == 2:
        self.pie_rule_available = False
    
    pocket = move
    while True:
      if self.testing and VISUALIZE_MOVES:
        print(self.board, pocket, self.current_player)
      # distribute pieces from the chosen pocket
      pieces = self.board[pocket]
      self.board[pocket] = 0
      for _ in range(pieces):
        pocket = pocket + 1 if pocket < 12 else 0
        self.board[pocket] += 1

      # if the last piece lands in the player's mancala, then go to the next step without changing turns
      if pocket == 6:
        return self.board[6] - self.board[13]
      
      if self.board[pocket] == 1:
        # if the last piece lands in an empty mancala on the player's side, then transfer the pieces in the opposite mancala to the player's mancala
        if pocket < 6:
          self.board[6] += self.board[12 - pocket] + 1
          self.board[pocket] = 0
          self.board[12 - pocket] = 0
        # regardless of whether the empty pocket is on the player's or opponent's side, the turn is over
        self.switch_turn()
        return self.board[13] - self.board[6]
  
  def switch_turn(self):
    self.current_player = not self.current_player
    self.turn += 1
    self.flip_board()

  def flip_board(self):
    self.board = np.concatenate([self.board[7:14], self.board[0:7]])

  def get_observation(self):
    observation = [0] * 288
    for i, pieces in enumerate(list(self.board[0:6]) + list(self.board[7:13])):
      observation[pieces + (i * 24)] = 1
    observation += [self.board[6], self.board[13], 1 if self.turn == 1 else 0]
    return observation

  def get_random_action(self):
    while True:
      random_move = random.randint(0, 6)
      if random_move == 6:
        if self.turn == 2 and self.pie_rule_available:
          return random_move
      elif self.board[random_move] != 0 or self.done:
        return random_move

  def get_best_action(self, values):
    max_value = -1
    best_action = None
    for i in range(len(values)):
      if values[i] > max_value:
        if i == 6:
          if self.turn == 2 and self.pie_rule_available:
            max_value = values[i]
            best_action = i
        elif self.board[i] != 0:
          max_value = values[i]
          best_action = i
    return best_action

  def reset(self):
    # Reset the state of the environment to an initial state
    self.board = np.array([4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0])
    self.current_player = bool(random.getrandbits(1))
    self.turn = 1
    self.pie_rule_available = True
    self.done = False
    self.used_pie_rule = False
    return self.get_observation()
  
  def render(self, mode='human', close=False):
    # "render" the environment to the screen
    print("Current player: " + str(self.current_player))
    print("Board: " + str(self.board))