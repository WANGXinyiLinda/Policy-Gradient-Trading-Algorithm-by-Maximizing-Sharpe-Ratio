from __future__ import print_function
import sys
import time
import random
import numpy as np
import pandas as pd
from math import floor
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, TensorBoard
from PG.model import *

class Agent(object):
    def __init__(self, data, train_frac = 0.8, traj_len = 20, batch_size = 4):
        self.traj_len = traj_len
        self.batch_size = batch_size
        self.num_train = int(train_frac*len(data))
        self.data_raw = np.array(data)
        self.data_scaled = self.scale(self.data_raw)
        self.num_val = (len(data) - self.num_train)//2
        self.num_val -= self.num_val % (self.traj_len*self.batch_size)
        self.num_test = len(data) - self.num_train - self.num_val
        self.num_test -= self.num_test % (self.traj_len*self.batch_size)
        print("number of train data: ", self.num_train, 
            "number of validation data: ", self.num_val,
            "number of test data: ", self.num_test)
        print("trajectory length is: ", self.traj_len)
        # data colume: ['Open', 'High', 'Low', 'Close', 'Volume'] -> [0,1,2,3,4]
        # (2019-01-31 01:20:00	to 2018-04-28 03:05:00)
        self.action_space = {0:-1, 1:1} # short and long
        self.num_actions = 2
        self.state_dim = 5
        self.model = build_model(self.state_dim, self.num_actions, self.traj_len, self.batch_size)
        self.logs_df = pd.DataFrame(columns = ['loss', 'lr', 'val_sharpe'])
    
    def scale(self, data):
        data_diff = np.concatenate([np.expand_dims(np.zeros(5), axis=0),
                                    np.diff(np.log(data), axis=0)], axis=0)
        train = data_diff[0:self.num_train,:]
        scaler = StandardScaler().fit(train)
        data_scaled = scaler.transform(data_diff)
        return data_scaled
    
    def sharpe_ratio(self, traj_idx, traj_actions):
        a = [self.action_space[i] for i in traj_actions]
        a = np.array(a[:-1])
        ret = np.diff(np.log(self.data_raw[traj_idx][:,3])) * a
        return np.mean(ret)/np.std(ret)

    def calculate_return(self, idx, a):
        return (self.data_raw[idx+1][3] - self.data_raw[idx][3]) * a
    
    def predict(self, traj_states):
        pred = self.model.predict(traj_states, batch_size=self.batch_size)
        traj_actions = np.argmax(pred, axis = -1)
        return traj_actions
    
    def get_a_batch_of_random_traj(self):
        X_batch = np.zeros((self.batch_size, self.traj_len, self.state_dim))
        idx_batch = np.zeros((self.batch_size, self.traj_len), dtype=int)
        for i in range(self.batch_size):
            begin_idx = np.random.randint(self.num_train - self.traj_len)
            idx_batch[i] = np.arange(begin_idx, begin_idx + self.traj_len, dtype = int)
            X_batch[i] = self.data_scaled[idx_batch[i]]
        return X_batch, idx_batch
    
    def train(self, num_epochs):
        steps_per_epoch = floor(self.num_train/self.traj_len/self.batch_size)
        print('Random sampling {} trajectors per epoch.'.format(steps_per_epoch))
        for epoch in range(num_epochs):
            history = []
            print("epoch {}...".format(epoch))
            for i in range(steps_per_epoch):
                X_batch, idx_batch = self.get_a_batch_of_random_traj()
                Y_batch = np.zeros((self.batch_size, self.traj_len, self.num_actions))
                if i < steps_per_epoch*(1 - epoch/num_epochs):
                    action_batch = np.random.randint(2, size = (self.batch_size, self.traj_len))     
                else:      
                    action_batch = self.predict(X_batch)
                for j in range(self.batch_size):
                    Y_batch[j][:][0] = self.sharpe_ratio(idx_batch[j], action_batch[j])
                loss = self.model.train_on_batch(X_batch, Y_batch) # PG
                lr = K.get_value(self.model.optimizer.lr)
                # record loss and learning rate
                history.append([loss, lr])
                print("step {}, loss: {}, sharpe ratio: {}".format(i, loss, np.mean(Y_batch[:,0,0])))
            self.model.save_weights('PG/logs/model_epoch{}.h5'.format(epoch))
            val_indices = np.arange(self.num_train, self.num_train + self.num_val).reshape((-1, self.traj_len))
            val_sharpe = self.eva(val_indices)
            history = list(np.mean(history, axis = 0))
            history.append(val_sharpe)
            print('epoch {}: loss: {}, learning rate: {}'.format(epoch, history[0], history[1]))
            df = pd.DataFrame([history], columns = ['loss', 'lr', 'val_sharpe'])
            self.logs_df = self.logs_df.append(df, ignore_index=True)
            self.logs_df.to_csv('PG/logs/history.csv')
    
    def eva(self, indices):
        states = self.data_scaled[indices]
        actions = self.predict(states)
        sharpe = self.sharpe_ratio(indices.flatten(), actions.flatten())
        print('validation sharpe ratio: {}'.format(sharpe))
        return sharpe

