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


class SNN:
    def __init__(self, N, T, V_rest, V_thresh, tau, ref_period, A_plus, A_minus, tau_stdp, w_init):
        # N: {total number of neurons in the network}
        self.N = N
        
        # T: {total timespan}
        self.T = T
        
        # V_rest: {resting potential}
        self.V_rest = V_rest
        
        # V_thresh: {firing threshold}
        self.V_thresh = V_thresh
        
        # tau: {the leaking efficient}
        # tau->1: less leaking, tau->0: more leaking
        self.tau = tau
        
        # ref_period: {refractory period}
        self.ref_period = ref_period
        
        # V[N][T]: the membrane potential for neuron_N, under the current timestamp T
        self.V = np.full((N, T), V_rest, dtype=float)
        
        # ref[N]: how much time the neuron_N is in its ref period
        self.ref = np.zeros(N)
        
        # last_spike[N]: the timestamp when neuron_N last fired, init to be -1000 indicating no firing before
        self.last_spike = np.full(N, -1000.0)
        
        # fired[N]: the indicator showing whether the neuron_N is fired or not in the current timestamp
        # =0: not fired, =1: fired
        self.fired = np.zeros(N, dtype=int)
        
        # W[M][N]: the weight between the neuron_M and neuron_N
        self.W = np.zeros((N, N))

        # W_history[M][N][T]: the weight between the neuron_M and neuron_N at time T
        self.W_history = np.zeros((N, N, T))
        
        # stim[N]: the cumulative input for neuron_N under the timestamp T
        self.stim = np.zeros((N, T))

        self.w_init = w_init

        self.A_plus = A_plus

        self.A_minus = A_minus

        self.tau_stdp = tau_stdp

        self.spike_times = [[] for _ in range(N)]


    def step(self, t):
        # I: the cumulative stimulation/input for each neuron under the current timestamp
        I = self.stim[:, t] + self.W.T @ self.fired

        # only update the membrane potential for active neurons (i.e., neurons that are not in the ref period)
        active = (self.ref == 0).astype(int)

        # update the membrane potential for all neurons
        # if a neuron is in the ref period, its membrane potential is just the resting potential
        self.V[:, t] = active * (self.tau * self.V[:, t-1] + I) + (1 - active) * self.V_rest

        # after the update, check fired or not
        self.fired = (self.V[:, t] >= self.V_thresh).astype(int)

        # if fired, the membrane potential resets to its resting potential
        # if not fired, no reset needed, keep the current state
        self.V[:, t] = (1 - self.fired) * self.V[:, t] + self.fired * self.V_rest

        # Visualization Trick (otherwise we are unable to capture the fire since the neuron resets to 0 immediately after firing)
        self.V[:, t] = np.where(self.fired, 200, self.V[:, t])

        self.ref -= 1
        self.ref = np.maximum(self.ref, 0)
        self.ref += self.fired * self.ref_period

        # STDP
        for i in range(self.N):
            if self.fired[i]:
                self.last_spike[i] = t
                self.spike_times[i].append(t)
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

        self.W_history[:, :, t] = self.W

    

    def run(self):
        for t in range(1, self.T):
            self.step(t)


    
    def plot(self):
        fig, axes = plt.subplots(self.N + 1, 1, figsize=(12, 2 * (self.N + 1)))
        
        for i in range(self.N):
            axes[i].plot(self.V[i])
            axes[i].set_title(f'E{i+1}')
            
        for i in range(self.N):
            for j in range(self.N):
                if self.W[i][j] != 0:
                    axes[self.N].plot(self.W_history[i][j], label=f'W{i}{j}')
        axes[self.N].legend()
        axes[self.N].set_title('Weights')
        
        plt.tight_layout()
        plt.show()