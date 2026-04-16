# PCN

This is a simple 0-1 implementation of a 3-neuron Predictive Coding Network.

This implementation uses binary spiking neurons and spike-timing-dependent plasticity (STDP) to explore ideas for building a learning network without traditional non-linear activation functions or backpropagation. Notably, under this network, learning happens on the fly and there is no separate training phase and no global loss function.

The future development will try to focus on scaling and learning on small datasets like MNIST.

## Update:
- Experimenting SNN to MNIST dataset $\Leftarrow$ Current Version
- Implemented vectorized SNN
