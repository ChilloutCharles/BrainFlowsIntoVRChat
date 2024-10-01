import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Layer, DepthwiseConv1D, SeparableConv2D , Conv1D
from keras.layers import Activation, Multiply, BatchNormalization, SpatialDropout1D, UpSampling1D, GlobalAveragePooling1D

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

# Noise Layer 
@keras.utils.register_keras_serializable()
class AddNoiseLayer(Layer):
    def __init__(self, noise_factor=0.1, **kwargs):
        super(AddNoiseLayer, self).__init__(**kwargs)
        self.noise_factor = noise_factor

    def call(self, inputs, training=None):
        if training:
            noise = self.noise_factor * tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=1.0)
            return inputs + noise
        return inputs

kernel = 3
e_rates = [1, 2, 4]
d_rates = list(reversed(e_rates))
act = 'elu'

def create_inner_layer(filters, kernel, dilation_rates, end_stride=1):
    # dialated depthwise convolves followed by pointwise convolve
    return Sequential(
        [DepthwiseConv1D(kernel, padding='same', dilation_rate=dr) for dr in dilation_rates] + 
        [Conv1D(filters, 1, padding='same', strides=end_stride)]
    )

@keras.utils.register_keras_serializable()
class Block(Layer):
    def __init__(self, filters, kernel, dilation_rates, strides=1, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.inner_layer = create_inner_layer(filters, kernel, dilation_rates, strides)
        self.residual = Conv1D(filters, 1, padding='same', strides=strides)

    def call(self, inputs):
        x = self.inner_layer(inputs)
        r = self.residual(inputs)
        return x + r

    def build(self, input_shape):
        super(Block, self).build(input_shape)

encoder = Sequential([
    Block(64, kernel, e_rates, 2),
    BatchNormalization(), Activation(act), # (80, 64)
    
    Block(32, kernel, e_rates, 2),
    BatchNormalization(), Activation(act), # (40, 32)
    
    Block(32, kernel, e_rates, 2),
    BatchNormalization(), Activation(act), # (20, 32)

    create_inner_layer(16, kernel, e_rates), Activation(act) # (20, 16)
])

decoder = Sequential([
    Block(16, kernel, d_rates),
    BatchNormalization(), Activation(act), UpSampling1D(2),
    
    Block(32, kernel, d_rates),
    BatchNormalization(), Activation(act), UpSampling1D(2),

    Block(32, kernel, d_rates),
    BatchNormalization(), Activation(act), UpSampling1D(2),
    
    create_inner_layer(64, kernel, d_rates), Activation('linear')
])

auto_encoder = Sequential([
    SpatialDropout1D(0.2),
    encoder,
    decoder
])

## First Layer to convert any channels to 64 ranged [0, 1]
def create_first_layer(chs=64):
    return Sequential([
        Conv1D(chs, 3, padding='causal', dilation_rate=1), Activation(act),
        Conv1D(chs, 3, padding='causal', dilation_rate=2), Activation(act), 
        Conv1D(chs, 3, padding='causal', dilation_rate=4), Activation('linear'),
    ])

## Last Layer to map latent space to custom classes
def create_last_layer(classes):
    return Sequential([
        GlobalAveragePooling1D(),
        Dense(classes, activation='softmax', kernel_regularizer='l2')
    ])