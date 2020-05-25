import gym
from env import MancalaEnv

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import Input
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

"""
class MancalaProcessor(Processor):
  def process_observation(self, observation):
    return observation
"""

env = MancalaEnv()

model = Sequential()
model.add(Flatten(input_shape=(1, 14)))
model.add(Dense(36, activation="relu"))
model.add(Dense(36, activation="relu"))
model.add(Dense(6, activation="relu"))

print(model.input_shape)

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps", value_max=1., value_min=0,
                              value_test=0, nb_steps=50000)

memory = SequentialMemory(limit=1000000, window_length=1)

dqn = DQNAgent(model=model, nb_actions=6, policy=policy, memory=memory, nb_steps_warmup=5000,
               gamma=.99, target_model_update=0.1, train_interval=5, delta_clip=1.)

dqn.compile(Adam(lr=0.001), metrics=["mae"])

callbacks = [ModelIntervalCheckpoint("weights_step_{step}.h5f", interval=10000)]
callbacks += [FileLogger("log.json", interval=100)]
dqn.fit(env, callbacks=callbacks, nb_steps=100000, log_interval=5000)

# After training is done, save the final weights
dqn.save_weights("weights_final.h5f", overwrite=True)

# Finally, evaluate our algorithm for 10 episodes.
dqn.test(env, nb_episodes=100, visualize=True)