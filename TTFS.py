import torch
import matplotlib.pyplot as plt

class TTFS:
    """
    Time-to-First-Spike encoder for 2D images.
    This implementation asuumes input images has been normalized to [0, 1].
    
    Args:
        delay_period: number of delays allowed per image column
        current: spike current amplitude
        threshold: minimum pixel value to generate a spike
    """

    def __init__(self, delay_period, current, threshold):
        self.delay_period = delay_period
        self.current = current
        self.threshold = threshold

    def encode(self, img):
        row = img.shape[0]
        col = img.shape[1]
        T = col * self.delay_period
        stim = torch.zeros(row, T)

        for c in range(col):
            for r in range(row):
                pixel = img[r, c].item()
                if pixel > 0.01: # if it is a black pixel, no spike
                    t = int(c * self.delay_period + (1 - pixel) * (self.delay_period - 1))
                    stim[r][t] = self.current

        return stim
    
    def plot(self, stim, label=None):
        plt.figure(figsize=(14, 4))
        plt.imshow(stim, aspect='auto', cmap='binary')
        plt.xlabel('Time step')
        plt.ylabel('Input neuron')
        if label is not None:
            title += f' (digit {label})'
        plt.title(title)
        plt.show()



