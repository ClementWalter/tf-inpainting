from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, LeakyReLU, Wrapper


def Boundless(*args, **kwargs):
    """
    Implement the base encoder for the discriminator model of
    [Boundless: Generative Adversarial Networks for Image Extension](https://arxiv.org/pdf/1908.07007.pdf)
    """
    return Sequential(
        [
            Conv2D(64, 5, strides=2, padding="same"),
            LeakyReLU(),
            Conv2D(128, 5, strides=2, padding="same"),
            LeakyReLU(),
            Conv2D(256, 5, strides=2, padding="same"),
            LeakyReLU(),
            Conv2D(256, 5, strides=2, padding="same"),
            LeakyReLU(),
            Conv2D(256, 5, strides=2, padding="same"),
            LeakyReLU(),
            Conv2D(256, 5, strides=2, padding="same"),
            LeakyReLU(),
            Conv2D(256, 5, strides=1, padding="valid"),
            LeakyReLU(),
            Flatten(),
        ],
        *args,
        **kwargs,
    )
