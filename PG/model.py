from __future__ import division

import six # Python 2 and 3 compatibility library
from keras.layers import (
    Input,
    Dense,
    LSTM,
    TimeDistributed
)
from keras.models import Model, Sequential
from keras.losses import mean_squared_error
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras import regularizers, optimizers
from keras.constraints import maxnorm
from keras import backend as K
import numpy as np

'''
y_true and y_pred does not have the same dimension.
y_pred: (num_actions, )
y_true: (num_actions, ). only need to ensure that y_true[0][0] = trajectory sharpe ratio
'''
def sharpe_loss(y_true, y_pred):
    a_sum = K.logsumexp(y_pred, axis=-1)
    a_max = K.max(y_pred, axis=-1)
    loss = K.batch_dot(y_true[:,:,0], a_sum - a_max)
    return loss

def build_model(state_dim, num_action, T, batch_size):
    inputs = Input(shape=(T, state_dim))
    policy_lstm = LSTM(500,
            return_sequences=True,
            batch_input_shape=(batch_size, T, state_dim), 
            use_bias=True,
            kernel_initializer='lecun_uniform',
            dropout=0.5,
            #kernel_constraint=maxnorm(3),
            #stateful=True,
            kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(inputs)
    raw_policy = TimeDistributed(Dense(num_action, activation = "linear"), name="raw_policy")(policy_lstm)

    model = Model(inputs=inputs, outputs=raw_policy)

    opt = optimizers.Adam(lr=1e-4) #opt = optimizers.RMSprop(lr=0.001) # default lr = 0.001
    model.compile(optimizer=opt, loss=sharpe_loss)
    return model