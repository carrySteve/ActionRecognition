# Baseline 1 & Baseline 2

- [Baseline 1 &amp; Baseline 2](#baseline-1-amp-baseline-2)
  - [Baseline 1 - Social LSTM](#baseline-1---social-lstm)
    - [Algorithm](#algorithm)
    - [Results](#results)

## Baseline 1 - Social LSTM

This baseline aims to implement *Alahi et al., Social LSTM: Human Trajectory Prediction in Crowded Spaces, CVPR 2016* on Stanford Drone Dataset
using PyTorch. The baselien aims to predict pedestrain's track with the help of Social Pooling Algorithm and LSTM (we replaced it with GRU)

Please note that a PyTorch version was not available then. Hence, I referred to a [Theano version](https://github.com/karthik4444/nn-trajectory-prediction "Social LSTM")

### Algorithm

Each pedestrain in the scene is assigned an RNN. And the hidden state of a pedestrain corresponds to the trajectory of a pedestrain's path thus far.

At each time step, a pooling layer gathers neighboring trajectories. If there are two or more people in the same grid, then their hidden states are pooled. Then, I embed this into a vector and feed it into the RNN.

![social-pooling](../imgs/social-pool.jpg)

### Results
