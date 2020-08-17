''' Q-Learning
   Off-policy TD(0) Control Reinforcement Learning Method '''

import numpy as np
import gym
from gym import wrappers, logger
import time
import matplotlib.pyplot as plt

class agent():
    def __init__(self, env='MountainCar-v0', bins=[20, 20], Min=[-1.2, -0.07], Max=[0.6, 0.07], actionSpace=[0, 1, 2], maxSteps=1000):
        
        self.env         = gym.make(env)
        self.actionSpace = actionSpace
        self.n_actions   = len(actionSpace)
        self.gamma       = 0.9
        self.eps         = 0.2
        self.minEps      = 0.01
        self.stateSpace  = []
        
        self.env._max_episode_steps = maxSteps

        for i, bin in enumerate(bins):
            self.stateSpace.append(np.linspace(Min[i], Max[i], bin))

        bins = np.array(bins) + 1
        bins = np.append(bins, self.n_actions)
        self.Q = np.zeros(bins)

    def load(self, path):
        self.Q = np.load(path)

    def save(self, path):
        np.save(path, self.Q)

    def get_state(self, obs):
        state = []
        for obsi, space in zip(obs, self.stateSpace):
            state.append(np.digitize(obsi, space))
        return state

    @staticmethod
    def get_state_action(state, action):
        state_action = state.copy()
        state_action.append(action)
        state_action = tuple(state_action)
        return state_action

    ''' Behavioural Policy
        epsilon-greedy policy'''
    def bPolicy(self, state):
        if np.random.random() < self.eps:
            return np.random.choice(self.actionSpace)
        return np.argmax(self.Q[tuple(state)])
    
    ''' Target Policy
        greedy policy'''
    def tPolicy(self, state):
        return np.argmax(self.Q[tuple(state)])

    def learn(self, n_episodes=10000, alpha=0.1, gamma=0.9, eps=1.0, decayRate=None, plot=True, path='./Q.npy'):
        self.n_episodes = n_episodes
        self.stepSize   = alpha
        self.gamma      = gamma
        self.eps        = eps
        self.decayRate  = decayRate if decayRate else 2 / self.n_episodes

        ploty = []
        y     = []
        for e in range(1, self.n_episodes+1):
            if not e%1000:
                self.save(path)
            done   = False
            obs    = self.env.reset()
            state  = self.get_state(obs)
            score  = 0
            while not done:
                action                = self.bPolicy(state)
                obs, reward, done, _  = self.env.step(action)
                # '''Uncomment these only for cart-pole environment'''
#                 if done:
#                     reward = -200
                score                += reward
                next_state            = self.get_state(obs)
                next_action           = self.tPolicy(next_state)
                state_action          = self.get_state_action(state, action)
                next_state_action     = self.get_state_action(next_state, next_action)
                '''Q(S, a) <- Q(S, A) + alpha[R + gamma*maxQ(S', a) - Q(S, A)]'''
                self.Q[state_action] += self.stepSize * (reward + (self.gamma * self.Q[next_state_action]) - self.Q[state_action])
                state                 = next_state
            self.eps -= self.decayRate if self.eps > self.minEps else 0
            print(f'[Episode, Score]: [{e}, {score}]')
            ploty.append(score)
            if not e % 100:
                y.append(sum(ploty) / 100)
                ploty = []
        
        self.env.close()
        self.save(path)
        
        if plot:
            plt.plot(y)
            plt.show()
        return y


    def test(self, n_episodes=10):
        for e in range(n_episodes):
            done   = False
            obs    = self.env.reset()
            state  = self.get_state(obs)
            score  = 0
            while not done:
                action                = self.tPolicy(state)
                obs, reward, done, _  = self.env.step(action)
                score                += reward
                next_state            = self.get_state(obs)
                self.env.render('human')
                time.sleep(0.02)
                state                 = next_state
            print(f'[Episode, Score]: [{e+1}, {score}]')
        self.env.close()

if __name__ == '__main__':
    qAgent = agent()
    qAgent.learn(n_episodes=2000)
    qAgent.load('./Q.npy')
    qAgent.test()


    # qAgent = agent(env='Acrobot-v1', bins=[10, 10, 10, 10, 40, 60], Min=[-1.0, -1.0, -1.0, -1.0, -4*np.pi, -9*np.pi], 
    #                Max=[1.0, 1.0, 1.0, 1.0, 4*np.pi, 9*np.pi], actionSpace=[0, 1, 2], maxSteps=1000)
    # qAgent.learn(n_episodes=50000, path='./Qa.npy')
    # qAgent.load('./Qa.npy')
    # qAgent.test()
    


    # qAgent = agent(env='CartPole-v0', bins=[20, 20, 20, 20], Min=[-0.209, -4, -2.4, -4], 
    #                Max=[0.209, 4, 2.4, 4], actionSpace=[0, 1], maxSteps=200)
    # qAgent.learn(n_episodes=10000, path='./Qc.npy')
    # qAgent.load('./Qc.npy')
    # qAgent.test()
