import gym
import random
env = gym.make('FrozenLake-v0')

env.reset()

print(env.observation_space) # which space the player is currently in
print(env.action_space) # actions the player can take


# 0 = left
# 1 = down
# 2 = right
# 3 = up

score = 0
numGames = 10000
numSteps = 10

# obs, rew, done, info = env.step(env.action_space.sample()) # take an action

def PlayGame():
    global score
    for i in range(numSteps):
        obs, rew, done, info = env.step(random.randint(1,2))
        env.render()
        if done:
            score += rew
            break

#Main Game Loop
for g in range (numGames):
    env.reset()
    PlayGame()

print (score)

