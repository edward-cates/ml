import gym
import os
import numpy as np
import sys

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 22})

# Environment initialization
folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'q_learning')
env = gym.wrappers.Monitor(gym.make('FrozenLake-v0'), folder, force=True)

# Q and rewards
Q = np.zeros((env.observation_space.n, env.action_space.n))
rewards = []
iterations = []

# Parameters
if len(sys.argv) == 1:
    alpha = 0.75
    discount = 0.95
else:
    alpha = float(sys.argv[1])
    discount = float(sys.argv[2])
#endif

episodes = 3000

max_r, max_e = 0, 0

old_p = [0,3,0,3,0,0,0,0,3,1,0,0,0,2,1,0]

# Episodes
for episode in xrange(episodes):
    # Refresh state
    state = env.reset()
    done = False
    t_reward = 0
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    # Run episode
    for i in xrange(max_steps):
        if done:
            break

        current = state
        action = np.argmax(Q[current, :] + np.random.randn(1, env.action_space.n) * (1 / float(episode + 1)))
        # action = old_p[current]

        state, reward, done, info = env.step(action)
        t_reward += (reward * 1)
        Q[current, action] += alpha * (reward + discount * np.max(Q[state, :]) - Q[current, action])

    rewards.append(t_reward)
    iterations.append(i)

    if (t_reward > max_r):
        max_r = t_reward
        max_e = episode
    #endif

# Close environment
env.close()

policy = np.argmax(Q, axis=1)

diff = 0
for i in range(16):
    if old_p[i] != policy[i]:
        diff += 1
    #endif
#endfor

ave = sum(rewards[-500:]) / 500

print('alpha: {}, discount: {}, episodes: {}, diff: {}, max_e: {}, max_r: {}, ave: {}'.format(alpha, discount, episodes, diff, max_e, max_r, ave))

# Plot results
def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

size = episodes / 25
chunks = list(chunk_list(rewards, size))
averages = [sum(chunk) / len(chunk) for chunk in chunks]

plt.plot(range(0, len(rewards), size), averages)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.ylim([0, 1])
# plt.show()

# Push solution
api_key = os.environ.get('GYM_API_KEY', False)
if api_key:
    print 'Push solution? (y/n)'
    if raw_input().lower() == 'y':
        gym.upload(folder, api_key=api_key)
