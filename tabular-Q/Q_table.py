import os
import numpy as np
from collections import deque

class Q_table(object):
    def __init__(self, env, discount = 0.95, lr = 0.01):
        self.Q = np.zeros((3,3,3,3,2))
        self.discount = discount
        self.lr = lr
        self.env = env
        self.hist_actions = deque(maxlen=self.env.episode_len)
        self.hist_idx = deque(maxlen=self.env.episode_len)
        self.Q_diff = np.zeros((3,3,3,3,2))
    
    def train(self, num_epochs):
        for i in range(num_epochs * (self.env.num_train//self.env.episode_len)):
            S, indices = self.env.get_a_ranodm_episode()
            print("epoch {}...".format(i))
            for j in range(self.env.episode_len-1):
                idx = indices[j]
                print("    episode {} with index {}...".format(j, idx))
                print(S[j])
                Q_s = self.Q[S[j][0]][S[j][1]][S[j][2]][S[j][3]]
                Q_s_1 = self.Q[S[j+1][0]][S[j+1][1]][S[j+1][2]][S[j+1][3]]
                a_idx = np.argmax(Q_s + np.random.rand()/float(1 + i + j/self.env.episode_len))
                a = self.env.action_space[a_idx]
                if i == 0:
                    r = self.env.calculate_return(idx, a)/10000
                else:
                    r = self.env.reward(idx, a, self.hist_idx, self.hist_actions)
                self.hist_actions.append(a)
                self.hist_idx.append(idx)
                
                Q_old = Q_s[a_idx]
                Q_s[a_idx] += self.lr * (r + self.discount * np.max(Q_s_1) - Q_s[a_idx])
                self.Q_diff[S[j][0]][S[j][1]][S[j][2]][S[j][3]][a_idx] = Q_s[a_idx] - Q_old

    def validation(self):
        S = self.env.get_states()
        A = []
        for i in range(self.env.num_val):
            Q_s = self.Q[S[i][0], S[i][1], S[i][2], S[i][3]]
            a_idx = np.argmax(Q_s)
            a = self.env.action_space[a_idx]
            A.append(a)
        sharpe = self.env.sharpe_ratio(range(self.env.num_train, self.env.num_train + self.env.num_val), A)
        return sharpe, A