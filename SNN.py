import numpy as np
import matplotlib.pyplot as plt

def chain(net, neurons):
    for i in range(len(neurons) - 1):
        net.W[neurons[i]][neurons[i+1]] = net.w_init

def fully_connect(net, neurons):
    for i in neurons:
        for j in neurons:
            if i != j:
                net.W[i][j] = net.w_init

def connect(net, src, targets):
    for t in targets:
        net.W[src][t] = net.w_init

def connect_layers(net, input_ids, hidden_ids, init_w=None):
    """
    Connect every neuron in input_ids to every neuron in hidden_ids.
    Use init_w if provided, otherwise use net.w_init.
    """
    w = init_w if init_w is not None else net.w_init
    for i in input_ids:
        for j in hidden_ids:
            net.W[i][j] = w


class SNN:
    """
    Spiking Neural Network with LIF neurons and STDP learning
 
    Args:
        N:          total number of neurons in the network
        T:          total timespan
        n_input:    number of input neurons 
                    (input-only, driven by stim, no LIF dynamics applied)
                    *** THE FIRST n_input NEURONS ARE INPUT NEURONS ***
        V_rest:     resting membrane potential
        V_thresh:   firing threshold
        tau:        leak coefficient
                    (tau->1: less leaking, tau->0: more leaking)
        ref_period: refractory period in time steps
        A_plus:     LTP learning rate
        A_minus:    LTD learning rate
        tau_stdp:   STDP time constant
        w_init:     initial neuron-connecting weight
        w_max:      maximum weight allowed
    """
    def __init__(self, N, T, n_input, V_rest, V_thresh, tau, ref_period, A_plus, A_minus, tau_stdp, w_init, w_max):
        self.N = N
        self.T = T
        self.n_input = n_input
        self.V_rest = V_rest
        self.V_thresh = V_thresh
        self.tau = tau
        self.ref_period = ref_period
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_stdp = tau_stdp
        self.w_init = w_init
        self.w_max = w_max
        
        self.V = np.full((N, T), V_rest, dtype=float) # V[N][T]: the membrane potential for neuron_N, under the current timestamp T
        self.ref = np.zeros(N) # ref[N]: how much time the neuron_N is in its ref period
        self.last_spike = np.full(N, -1000.0) # last_spike[N]: the timestamp when neuron_N last fired, init to be -1000 indicating no firing before
        self.fired = np.zeros(N, dtype=int) # fired[N]: the indicator showing whether the neuron_N is fired or not in the current timestamp (=0: not fired, =1: fired)
        self.W = np.zeros((N, N)) # W[M][N]: the weight between the neuron_M and neuron_N
        self.stim = np.zeros((N, T)) # stim[N]: the cumulative input for neuron_N under the timestamp T
        self.spike_times = [[] for _ in range(N)]
        self.W_history = np.zeros((N, N, T)) # weight history, intended only for non-zero connections



    def step(self, t):
        # UNDER THE CURRENT TIMESTAMP T, WE DO:

        # STEP 1: COMPUTE THE CUMULATIVE INPUT FOR ALL NEURONS
        I = self.stim[:, t] + self.W.T @ self.fired

        # STEP 2: UPDATE INPUT NEURONS
        # Input neurons fire when they receive a stimulation
        for i in range(self.n_input):
            if self.ref[i] > 0:
                self.ref[i] -= 1
                self.fired[i] = 0
            elif self.stim[i, t] > 0:
                self.fired[i] = 1
                self.last_spike[i] = t
                self.spike_times[i].append(t)
                self.ref[i] = self.ref_period
                self.V[i, t] = self.stim[i, t]  # for visualization
            else:
                self.fired[i] = 0
                self.V[i, t] = 0
        
        # STEP 3: UPDATE HIDDEN NEURONS
        # Hidden neurons follow LIF dynamics to decay or fire
        h_start = self.n_input
        h = slice(h_start, self.N)

        active = (self.ref[h] == 0).astype(int) # only update the membrane potential for active neurons (i.e., neurons that are not in the ref period)
        
        # update the membrane potential for all neurons
        # if a neuron is in the ref period, its membrane potential is just the resting potential
        self.V[h, t] = active * (self.tau * self.V[h, t-1] + I[h]) + (1 - active) * self.V_rest

        # after the update, check fired or not
        h_fired = (self.V[h, t] >= self.V_thresh).astype(int)
        self.fired[h] = h_fired

        # if fired, the membrane potential resets to its resting potential
        # if not fired, no reset needed, keep the current state
        self.V[h, t] = (1 - self.fired[h]) * self.V[h, t] + self.fired[h] * self.V_rest

        # Visualization Trick (otherwise we are unable to capture the fire since the neuron resets to 0 immediately after firing)
        self.V[h, t] = np.where(self.fired[h], self.V_thresh * 1.5, self.V[h, t])

        self.ref[h] -= 1
        self.ref[h] = np.maximum(self.ref[h], 0)
        self.ref[h] += self.fired[h] * self.ref_period

        # Record spike times for hidden neurons
        for i in range(h_start, self.N):
            if self.fired[i]:
                self.last_spike[i] = t
                self.spike_times[i].append(t)

        # STEP 4: STDP (for fired hidden neurons only)
        for i in range(h_start, self.N):
            if self.fired[i]:
                # the neuron_I is a post neuron: DO LTP
                for j in range(self.N):
                    if self.W[j][i] > 0:
                        dt_stdp = t - self.last_spike[j]
                        if dt_stdp > 0:
                            self.W[j][i] += self.A_plus * np.exp(-dt_stdp / self.tau_stdp)
                # the neuron_I is a pre neuron: DO LTD
                for k in range(self.N):
                    if self.W[i][k] > 0:
                        dt_stdp = self.last_spike[k] - t
                        if dt_stdp < 0:
                            self.W[i][k] -= self.A_minus * np.exp(dt_stdp / self.tau_stdp)
        
        # Weight clipping
        self.W = np.clip(self.W, 0, self.w_max)

        # Record weight history
        self.W_history[:, :, t] = self.W
    


    def run(self):
        for t in range(1, self.T):
            self.step(t)


    
    def plot(self, neuron_ids=None):
        if neuron_ids is None:
            neuron_ids = range(self.N)
        
        n_plots = len(neuron_ids) + 1  # +1 for weights
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots))
        
        if n_plots == 2:
            axes = [axes]
        
        for idx, neuron_id in enumerate(neuron_ids):
            axes[idx].plot(self.V[neuron_id])
            label = 'Input' if neuron_id < self.n_input else 'Hidden'
            axes[idx].set_title(f'{label} Neuron {neuron_id}')
 
        # Plot non-zero weights
        ax_w = axes[-1]
        for i in range(self.N):
            for j in range(self.N):
                if np.any(self.W_history[i, j, :] != 0):
                    ax_w.plot(self.W_history[i, j, :], label=f'W{i}->{j}')
        ax_w.set_title('Weights')
        if ax_w.get_legend_handles_labels()[1]:  # only add legend if there are labels
            ax_w.legend(fontsize=6, ncol=4)
 
        plt.tight_layout()
        plt.show()