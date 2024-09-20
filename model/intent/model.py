import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Layer, DepthwiseConv2D, SeparableConv2D , Conv1D, UpSampling2D
from keras.layers import Activation, Flatten, Multiply, BatchNormalization, Dropout

## Spatial Attention (Thanks Summer!)
@keras.saving.register_keras_serializable()
class SpatialAttention(Layer):
    def __init__(self, classes, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.classes = classes
        self.conv1 = Conv1D(self.classes, self.kernel_size, padding='same', activation='silu', use_bias=False)
        self.conv2 = Conv1D(1, self.kernel_size, padding='same', activation='sigmoid', use_bias=False)
    
    def build(self, input_shape):
        super(SpatialAttention, self).build(input_shape)
    
    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        x = tf.concat([avg_out, max_out], axis=2)
        x = self.conv1(x)
        x = self.conv2(x)
        return Multiply()([inputs, x])

## Encoder and Decoder Trained on the physionet motor imagery dataset
## https://www.physionet.org/content/eegmmidb/1.0.0/
## Thanks again to Summer, Programmerboi, Hosomi
## Modification to follow along this paper
## https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7522466/

@keras.saving.register_keras_serializable()
class ExpandDimsLayer(Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=-1)

@keras.saving.register_keras_serializable()
class SqueezeDimsLayer(Layer):
    def call(self, inputs):
        return tf.squeeze(inputs, axis=-1)

kernel = (3, 3)
act = 'leaky_relu'

encoder = Sequential([
    ExpandDimsLayer(),
    SeparableConv2D(16, kernel, padding='same'),
    DepthwiseConv2D(kernel, padding='same', strides=2),
    BatchNormalization(), Activation(act),
    
    SeparableConv2D(16, kernel, padding='same'),
    DepthwiseConv2D(kernel, padding='same', strides=2),
    BatchNormalization(), Activation(act),

    SeparableConv2D(16, kernel, padding='same'),
    DepthwiseConv2D(kernel, padding='same', strides=2),
    BatchNormalization(), Activation(act),

    SeparableConv2D(16, kernel, padding='same'),
])

decoder = Sequential([
    SeparableConv2D(16, kernel, padding='same'),
    BatchNormalization(), Activation(act), UpSampling2D(2),
    
    SeparableConv2D(16, kernel, padding='same'),
    BatchNormalization(), Activation(act), UpSampling2D(2),

    SeparableConv2D(16, kernel, padding='same'),
    BatchNormalization(), Activation(act), UpSampling2D(2),
    
    SeparableConv2D(1, kernel, padding='same'), Activation('relu'),
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