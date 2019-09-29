<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>

# Policy Gradient Trading Algorithm by Maximizing Sharpe Ratio

This is the sencod half of my capstone project and the first half can be seen [here](https://github.com/WANGXinyiLinda/Deep-Q-Learning-Bitcoin-Trading-Agent), which is a trading algorithm using deep Q learning. This project is the second half of my capstone project. In this project, I designed a trading algorithm using policy gradient to maximize the profit while incorporating the risk factor by directly maximizing the Sharpe ratio over a fixed period of time. Then I conduct experiments on a Bitcoin dataset to compare its performance with a Q learning algorithm. More details can be found in my [final report](SCIE4500_Final_Report.pdf) and in my final presentation [slides](SCIE4500_Final_presentation.pdf).

## Action space:

    {LONG=1, SHORT=-1}

## State:

I define the state variable $s_t$ as a concatenation of the five-tuple discribing the current market $x_t$ and the previous hidden state from LSTM $h_{t-1}$. For more details, see section 2.2 in my final report.

## Policy network:

Map curren state $s_t$ to the policy $\pi(\text(long)|s_t)$ and $\pi(\text(short)|s_t)$. A illustration of the overall model structure is shown below.

![](img/model.png)

## Usage:

    For the proposed policy gradient algorithm, please see the [jupyter notebook](PG/PG.ipynb) in the ./PG directory.

    For the Q learning algorithm baseline, please see the [jupyter notebook](tabular-Q/tabular-Q.ipynb) in the ./tabular-Q directory.

## Test result:

Blow are the cumulative profite by using the proposed policy gradient algorithm and the Q learning algorithm respectively.

![Propsed policy gradient algorithm](img/pg_100.png)
![Q learning algorithm](img/Q_100.png)
