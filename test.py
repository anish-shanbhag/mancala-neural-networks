from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import Input
from keras.optimizers import Adam

from env import MancalaEnv

from rl.agents.dqn import DQNAgent

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

model = Sequential()
model.add(Flatten(input_shape=(1, 14)))
model.add(Dense(36, activation="relu"))
model.add(Dense(36, activation="relu"))
model.add(Dense(6, activation="relu"))

memory = SequentialMemory(limit=1000000, window_length=1)

agent = DQNAgent(model=model, nb_actions=6, memory=memory)
agent.target_model = model

agent.load_weights("weights_final.h5f")
agent.compile(Adam(lr=.00025), metrics=["mae"])

env = MancalaEnv()
agent.test(env, nb_episodes=1000, visualize=False)