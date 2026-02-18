import numpy as np

class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) Neuron Model.
    Core unit of Neuromorphic Computing (Intel Lava simulation).
    """
    def __init__(self, threshold=1.0, decay=0.9, rest=0.0):
        self.threshold = threshold
        self.decay = decay
        self.rest = rest
        self.voltage = rest
        self.spike_log = []

    def step(self, current_input):
        """
        Process one time step.
        Returns: 1 if spike, 0 otherwise.
        """
        # Integrate
        self.voltage = self.voltage * self.decay + current_input

        # Fire
        spike = 0
        if self.voltage >= self.threshold:
            spike = 1
            self.voltage = self.rest # Reset

        self.spike_log.append(spike)
        return spike

class SpikingLayer:
    def __init__(self, size):
        self.neurons = [LIFNeuron() for _ in range(size)]

    def forward(self, input_currents):
        """
        input_currents: Array of currents for each neuron.
        Returns: Array of spikes.
        """
        return [n.step(c) for n, c in zip(self.neurons, input_currents)]
