# Policy Gradient Trading Algorithm by Maximizing Sharpe Ratio

This is the second half of my capstone project and the first half can be seen [here](https://github.com/WANGXinyiLinda/Deep-Q-Learning-Bitcoin-Trading-Agent), which is a trading algorithm using deep Q learning. In this project, I designed a trading algorithm using policy gradient to maximize the profit while incorporating the risk factor by directly maximizing the Sharpe ratio over a fixed period of time. Then I conduct experiments on a Bitcoin dataset to compare its performance with a Q learning algorithm. More details can be found in my [final report](SCIE4500_Final_Report.pdf) and in my final presentation [slides](SCIE4500_Final_presentation.pdf).

## Action space:

    {LONG=1, SHORT=-1}

## State:

I define the state variable ![](https://latex.codecogs.com/gif.latex?s_t) as a concatenation of the five-tuple discribing the current market ![](https://latex.codecogs.com/gif.latex?x_t) and the previous hidden state from LSTM ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D). For more details, see section 2.2 in my final report.

## Policy network:

The policy network maps the curren state ![](https://latex.codecogs.com/gif.latex?s_t) to the policy ![](https://latex.codecogs.com/gif.latex?%5Cpi%28%5Ctext%28long%29%7Cs_t%29) and ![](https://latex.codecogs.com/gif.latex?%5Cpi%28%5Ctext%28short%29%7Cs_t%29). An illustration of the overall model structure is shown below.

![](img/model.png)

## Usage:

This python 3 implementation is mostly based on Keras with a Tensorflow backend. You will also need pandas, numpy and sklearn to handle the data.

For the proposed policy gradient algorithm, please see the jupyter notebook in the ./PG directory.

For the Q learning algorithm baseline, please see the jupyter notebook in the ./tabular-Q directory.

## Test result:

Below are the cumulative profits by using the proposed policy gradient algorithm (left) and the Q learning algorithm (right) respectively.

![Propsed policy gradient algorithm](img/pg_100.png)
![Q learning algorithm](img/Q_100.png)
