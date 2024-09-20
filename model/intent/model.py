import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Layer, DepthwiseConv2D, SeparableConv2D , Conv1D, UpSampling2D, MaxPooling2D
from keras.layers import Activation, Flatten, Multiply, BatchNormalization, Dropout

## Spatial Attention (Thanks Summer!)
@keras.saving.register_keras_serializable()
class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.attn = SeparableConv2D(1, kernel_size, padding='same', activation='sigmoid')
    
    def build(self, input_shape):
        super(SpatialAttention, self).build(input_shape)
    
    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        x = tf.concat([avg_out, max_out], axis=-1)
        x = self.attn(x)
        return Multiply()([inputs, x])

## Encoder and Decoder Trained on the physionet motor imagery dataset
## https://www.physionet.org/content/eegmmidb/1.0.0/
## Thanks again to Summer, Programmerboi, Hosomi
## Modification to follow along this paper
## https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-020-00203-9

@keras.saving.register_keras_serializable()
class ExpandDimsLayer(Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=-1)

@keras.saving.register_keras_serializable()
class SqueezeDimsLayer(Layer):
    def call(self, inputs):
        return tf.squeeze(inputs, axis=-1)

kernel = (3, 3)
e_rates = [1, 2]
d_rates = list(reversed(e_rates))
act = 'leaky_relu'

def create_block(filters, kernel, dilation_rates, end_stride=1):
    # dialated depthwise convolves followed by pointwise convolve
    return Sequential(
        [DepthwiseConv2D(kernel, padding='same', dilation_rate=dr) for dr in dilation_rates] + 
        [SeparableConv2D(filters, 1, padding='same', strides=end_stride)]
    )

encoder = Sequential([
    ExpandDimsLayer(),
    create_block(16, kernel, e_rates, 2),
    BatchNormalization(), Activation(act),
    
    create_block(16, kernel, e_rates, 2),
    BatchNormalization(), Activation(act),
    
    create_block(16, kernel, e_rates, 2),
    BatchNormalization(), Activation(act),

    create_block(16, kernel, e_rates), Activation(act)
])

decoder = Sequential([
    create_block(16, kernel, d_rates),
    BatchNormalization(), Activation(act), UpSampling2D(2),
    
    create_block(16, kernel, d_rates),
    BatchNormalization(), Activation(act), UpSampling2D(2),

    create_block(16, kernel, d_rates),
    BatchNormalization(), Activation(act), UpSampling2D(2),
    
    create_block(1, kernel, d_rates), Activation('relu'),
    SqueezeDimsLayer()
])

auto_encoder = Sequential([
    encoder,
    decoder
])

## First Layer to convert any channels to 64 ranged [0, 1]
def create_first_layer(chs=64):
    return Sequential([
        Conv1D(chs//4, 3, padding='same'),
        BatchNormalization(), Activation('silu'),
        Conv1D(chs//2, 3, padding='same'),
        BatchNormalization(), Activation('silu'), 
        Conv1D(chs//1, 3, padding='same'),
        BatchNormalization(), Activation('sigmoid'), 
    ])

## Last Layer to map latent space to custom classes
def create_last_layer(classes):
    return Sequential([
        Flatten(),
        Dropout(0.1),
        Dense(classes, activation='softmax', kernel_regularizer='l2')
    ])