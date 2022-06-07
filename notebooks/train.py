import tensorflow as tf
import tensorflow_gan as tfgan


def Generator(input_shape):
    """
    Generator model for inpainting.

    Args:
        input_shape: should be a 3 channels tensors Vx, Vy, Pressure
    """
