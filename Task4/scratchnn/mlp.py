"""Simple MLP built from Value objects: Neuron, Layer, MLP."""
import random
from typing import List
from .value import Value


class Neuron:
    def __init__(self, nin, activation='tanh', weight_init='normal'):
        self.w = [Value(random.uniform(-1, 1) * (1.0 / (nin ** 0.5)) ) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x, training=True, dropout_p=0.0):
        act = sum((wi * xi) for wi, xi in zip(self.w, x)) + self.b
        if self.activation == 'tanh':
            out = act.tanh()
        elif self.activation == 'relu':
            out = act.relu()
        else:
            out = act

        # dropout (inverted dropout)
        if training and dropout_p > 0.0:
            if random.random() < dropout_p:
                return Value(0.0)
            else:
                return out * (1.0 / (1.0 - dropout_p))

        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, activation='tanh', dropout_p=0.0):
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]
        self.dropout_p = dropout_p

    def __call__(self, x, training=True):
        out = [n(x, training=training, dropout_p=self.dropout_p) for n in self.neurons]
        return out

    def parameters(self):
        params = []
        for n in self.neurons:
            params += n.parameters()
        return params


class MLP:
    def __init__(self, nin, nouts: List[int], activation='tanh', dropout_p=0.0):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            act = activation if i < len(nouts) - 1 else None
            self.layers.append(Layer(sz[i], sz[i+1], activation=act, dropout_p=dropout_p if i < len(nouts)-1 else 0.0))

    def __call__(self, x, training=True):
        out = [Value(xx) for xx in x]
        for layer in self.layers:
            out = layer(out, training=training)
        return out

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
