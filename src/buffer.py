import tensorflow as tf
import numpy as np


class Buffer:

    def __init__(self, capacity=None, batch_size=32):
        self.memory = []
        self.batch_size = batch_size
        if isinstance(capacity, int) and capacity > 0:
            self.capacity = capacity
        else:
            self.capacity = float('inf')

    def __len__(self) -> int:
        return len(self.memory)

    def add(self, transition) -> None:
        self.memory.append(transition)
        if len(self) > self.capacity:
            self.memory.pop(0)

    def sample(self) -> tuple:
        indices = np.random.randint(len(self), size=self.batch_size)
        transitions = [self.memory[index] for index in indices]
        transitions = list(map(tf.convert_to_tensor, zip(*transitions)))
        return transitions