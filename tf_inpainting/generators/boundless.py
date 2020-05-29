import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Concatenate, Input, UpSampling2D, Lambda

from tf_inpainting.layers.gated_layer import Gated


def Boundless(input_shape):
    """
    Implement the generator model of [Boundless: Generative Adversarial Networks for Image Extension](https://arxiv.org/pdf/1908.07007.pdf)
    """
    input_tensor = Input(input_shape)
    encoder = Sequential(
        [
            Gated(Conv2D)(32, 5, activation="elu", padding="same"),
            Gated(Conv2D)(64, 3, strides=2, activation="elu", padding="same"),
            Gated(Conv2D)(64, 3, activation="elu", padding="same"),
            Gated(Conv2D)(128, 3, strides=2, activation="elu", padding="same"),
            Gated(Conv2D)(128, 3, activation="elu", padding="same"),
            Gated(Conv2D)(128, 3, activation="elu", padding="same"),
            Gated(Conv2D)(128, 3, dilation_rate=2, activation="elu", padding="same"),
            Gated(Conv2D)(128, 3, dilation_rate=4, activation="elu", padding="same"),
            Gated(Conv2D)(128, 3, dilation_rate=8, activation="elu", padding="same"),
            Gated(Conv2D)(128, 3, dilation_rate=16, activation="elu", padding="same"),
            Gated(Conv2D)(128, 3, activation="elu", padding="same"),
        ]
    )
    output = encoder(input_tensor)
    output = Concatenate()([output, encoder.layers[4].output])
    output = Gated(Conv2D)(128, 3, activation="elu", padding="same")(output)
    output = Concatenate()([output, encoder.layers[3].output])
    output = UpSampling2D(size=2, interpolation="bilinear")(output)
    output = Gated(Conv2D)(64, 3, activation="elu", padding="same")(output)
    output = Concatenate()([output, encoder.layers[2].output])
    output = Gated(Conv2D)(64, 3, activation="elu", padding="same")(output)
    output = Concatenate()([output, encoder.layers[1].output])
    output = UpSampling2D(size=2, interpolation="bilinear")(output)
    output = Gated(Conv2D)(32, 3, activation="elu", padding="same")(output)
    output = Concatenate()([output, encoder.layers[0].output])
    output = Gated(Conv2D)(16, 3, activation="elu", padding="same")(output)
    output = Conv2D(3, 3, padding="same")(output)
    output = Lambda(lambda x: tf.clip_by_value(x, -1, 1))(output)
    return Model(input_tensor, output)
