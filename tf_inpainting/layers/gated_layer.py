from functools import wraps
from tensorflow.keras.layers import Layer, Multiply


class GatedLayer(Layer):
    """
    A gated layer is layer that learns a mask over its standard output. The mask is learnt by creating another instance of the gated layer
    with an enforced sigmoid activation function. Then the two outputs are multiplied.
    """

    def __init__(self, layer_class, *args, **kwargs):
        super().__init__()
        self.conv = layer_class(*args, **kwargs)
        self.conv_gate = layer_class(*args, **{**kwargs, "activation": "sigmoid"})
        self.multiply = Multiply()

    def call(self, inputs, **kwargs):
        return self.multiply([self.conv(inputs), self.conv_gate(inputs)])


class Gated:
    """
    A class decorator to be used to "gate" a given class. For example, use: Gated(Conv2D)(*args, **kwargs).
    """

    def __init__(self, layer_class):
        self.layer_class = layer_class

    @wraps(Layer.__init__)
    def __call__(self, *args, **kwargs):
        return GatedLayer(self.layer_class, *args, **kwargs)
