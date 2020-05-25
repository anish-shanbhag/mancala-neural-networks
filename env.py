import gym
from gym import spaces
import numpy as np
import random

TESTING = True
RANDOM = False
VISUALIZE_MOVES = False

class MancalaEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(MancalaEnv, self).__init__()
    self.games_won = 0
    self.games = 0
    self.reward_range = (-100, 100)
    self.action_space = spaces.Discrete(6)
    self.observation_space = spaces.Box(low=0, high=50, shape=(14,), dtype=np.uint8)

  def step(self, move):

    # Execute one time step within the environment
    self.current_step += 1
    if TESTING and not RANDOM:
      print("-------------------------------")
      print(self.board[7:14][::-1])
      print(" ", self.board[0:7])
      print("-------------------------------")
    if self.current_player or not TESTING:
      reward = self.move(move)
    else:
      reward = 0
      while True:
        random_move = random.randint(0, 6) if RANDOM else int(input("Choose move: "))
        if self.board[random_move] != 0:
          break
      self.move(random_move)
      
    # if one side is completely empty, then the game is over - transfer remaining pieces to the corresponding player's mancala
    if np.all(self.board[0:6] == 0) or np.all(self.board[7:13] == 0):
      mancala_to_fill = 13 if np.all(self.board[0:6] == 0) else 6
      self.board[mancala_to_fill] += np.sum(np.concatenate([self.board[0:6], self.board[7:13]]))
      self.done = True
      
      player_mancala = 6 if self.current_player else 13
      opponent_mancala = 13 if self.current_player else 6
      
      if False:
        print("\nPlayer total: " + str(self.board[player_mancala]))
        print("Opponent total: " + str(self.board[opponent_mancala]))

      self.games += 1
      if self.board[player_mancala] > self.board[opponent_mancala]:
        self.games_won += 1
      print("Win percentage: ", self.games_won / self.games)
      
    
    observation = self.board
    return observation, reward, self.done, {}

  def move(self, move):
    pieces = self.board[move]
    old_pieces = self.board[6]

    # invalid move, negative reward
    if pieces == 0:
      # self.switch_turn()
      self.done = True
      # print("Invalid move")
      return -10000
    
    pocket = move
    while True:
      if TESTING and VISUALIZE_MOVES:
        print(self.board, pocket)
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
        reward = self.board[6] - self.board[13]
        self.switch_turn()
        return reward
  
  def switch_turn(self):
    self.current_player = not self.current_player
    self.board = np.concatenate([self.board[7:14], self.board[0:7]])

  def reset(self):
    # Reset the state of the environment to an initial state
    self.current_step = 0
    self.board = np.array([4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0])
    self.current_player = bool(random.getrandbits(1))
    self.done = False
    self.invalid_move = False
    return self.board
  
  def render(self, mode='human', close=False):
    # "render" the environment to the screen
    print("Current step: " + str(self.current_step))
    print("Current player: " + str(self.current_player))
    print("Board: " + str(self.board))