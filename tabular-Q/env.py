import os
import numpy as np
from sklearn.preprocessing import StandardScaler

class Market():
    def __init__(self, data, val_split = 0.1, past_steps = 60, episode_len = 50,
                deviate = 0.425, bars = [60, 60*24, 7*60*24]):
        self.past_steps = past_steps
        self.deviate = deviate
        self.bars = bars
        self.num_train = int((1-val_split)*len(data))
        self.data_raw = np.array(data)
        self.data = self.scale(self.data_raw)
        self.num_val = len(self.data) - self.num_train
        print("number of train data: ", self.num_train, "number of validation data: ", self.num_val)
        # data colume: ['Open', 'High', 'Low', 'Close', 'Volume'] -> [0,1,2,3,4]
        # data index: 'Close time' (2019-01-31 01:20:00	to 2018-04-28 03:05:00)
        self.action_space = {0:-1, 1:1} # short and long
        self.episode_len = episode_len
    
    def scale(self, data):
        train = data[0:self.num_train,:]
        scaler = StandardScaler().fit(train)
        return scaler.transform(data)
    
    def get_state_idx(self, idx, bar, col_idx):
        window = self.data[idx-bar+1:idx][:, col_idx]
        value = self.data[idx][col_idx]
        mean = np.mean(window)
        std = np.std(window)
        if value > mean + self.deviate*std:
            return 2
        elif value < mean - self.deviate*std:
            return 0
        else:
            return 1

    def sharpe_ratio(self, hist_idx, hist_actions):
        a = np.array(hist_actions)
        ret = np.diff(np.log(self.data_raw[hist_idx][:,3])) * a[:-1]
        return np.mean(ret)/np.std(ret)

    def calculate_return(self, idx, a):
        return (self.data_raw[idx+1][3] - self.data_raw[idx][3]) * a

    def reward(self, idx, a, hist_idx, hist_actions):
        past_sharpe = self.sharpe_ratio(hist_idx, hist_actions)
        immd_return = self.calculate_return(idx, a)
        return past_sharpe * immd_return

    def get_a_ranodm_episode(self):
        begin_idx = np.random.randint(self.num_train - self.episode_len)
        indices = range(begin_idx, begin_idx + self.episode_len)
        S = np.zeros((self.episode_len, 4), dtype = int)
        for idx in indices:
            S[idx-begin_idx][0] = self.get_state_idx(idx, self.bars[0], 3)
            S[idx-begin_idx][1] = self.get_state_idx(idx, self.bars[1], 3)
            S[idx-begin_idx][2] = self.get_state_idx(idx, self.bars[2], 3)
            S[idx-begin_idx][3] = self.get_state_idx(idx, self.bars[1], 4)
        return S, indices
    
    def get_states(self):
        S = np.zeros((self.num_val, 4), dtype = int)
        for idx in range(self.num_train, self.num_train + self.num_val):
            S[idx-self.num_train][0] = self.get_state_idx(idx, self.bars[0], 3)
            S[idx-self.num_train][1] = self.get_state_idx(idx, self.bars[1], 3)
            S[idx-self.num_train][2] = self.get_state_idx(idx, self.bars[2], 3)
            S[idx-self.num_train][3] = self.get_state_idx(idx, self.bars[1], 4)
        return S