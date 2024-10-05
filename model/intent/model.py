import tensorflow as tf
import numpy as np
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Layer, DepthwiseConv1D, SeparableConv2D , Conv1D, Attention
from keras.layers import Activation, Multiply, BatchNormalization, SpatialDropout1D, UpSampling1D, GlobalAveragePooling1D, Dropout, Input, LayerNormalization, Flatten
from keras.losses import MeanSquaredError as MSE, CategoricalCrossentropy, CosineSimilarity

## Spatial Attention (Thanks Summer!)
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

@keras.utils.register_keras_serializable()
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.attn = Attention()
    
    def call(self, x):
        return self.attn([x, x])

    def build(self, input_shape):
        super(SelfAttention, self).build(input_shape)

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

## Encoder and Decoder Trained on the physionet motor imagery dataset
## https://www.physionet.org/content/eegmmidb/1.0.0/
## Thanks again to Summer, Programmerboi, Hosomi

kernel = 2
e_rates = [1, 2, 4]
d_rates = list(reversed(e_rates))
act = 'elu'
rank = 8  # Low-rank dimension for LoRA

## LoRA layers for use in downstream classification
## https://arxiv.org/abs/2106.09685
@keras.utils.register_keras_serializable()
class LoRALayer(Layer):
    def __init__(self, rank, **kwargs):
        super(LoRALayer, self).__init__(**kwargs)
        self.rank = rank
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.A = self.add_weight(shape=(input_dim, self.rank),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='lora_A')
        self.B = self.add_weight(shape=(self.rank, input_dim),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='lora_B')
        super(LoRALayer, self).build(input_shape)

    def call(self, inputs):
        lora_update = tf.matmul(inputs, self.A)
        lora_update = tf.matmul(lora_update, self.B)
        return inputs + lora_update

# activates LoRA layers for downstream 
def partial_trainable(seq_model):
    for layer in seq_model.layers:
        if not isinstance(layer, LoRALayer):
            layer.trainable = False  
        else:
            layer.trainable = True

## Modification of seperable convolutions to follow along this paper
## https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-020-00203-9
@keras.utils.register_keras_serializable()
class StackedDepthSeperableConv1D(Layer):
    def __init__(self, filters, kernel_size, dilation_rates, stride=1, use_residual=False, **kwargs):
        super(StackedDepthSeperableConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.dilation_rates = dilation_rates
        self.depthwise_stack = Sequential([DepthwiseConv1D(kernel_size, padding='same', dilation_rate=dr) for dr in dilation_rates])
        self.pointwise_conv = Conv1D(filters, 1, padding='same', strides=stride)
        self.residual_conv = None
        if use_residual:
            self.residual_conv = Conv1D(filters, 1, padding='same', strides=stride)
    
    def call(self, inputs):
        depthwise_output = self.depthwise_stack(inputs)
        output = self.pointwise_conv(depthwise_output)
        if self.residual_conv:
            output += self.residual_conv(inputs)
        return output
    
    def build(self, input_shape):
        super(StackedDepthSeperableConv1D, self).build(input_shape)

encoder = Sequential([
    StackedDepthSeperableConv1D(64, kernel, e_rates, 2, True),
    BatchNormalization(), Activation(act), # (80, 64)
    LoRALayer(rank), 
    
    StackedDepthSeperableConv1D(32, kernel, e_rates, 2, True),
    BatchNormalization(), Activation(act), # (40, 32)
    LoRALayer(rank), 
    
    StackedDepthSeperableConv1D(32, kernel, e_rates, 2, True),
    BatchNormalization(), Activation(act), # (20, 32)
    LoRALayer(rank), 

    StackedDepthSeperableConv1D(16, kernel, e_rates, 1, False), # (20, 16)
    Activation('linear')
])

decoder = Sequential([
    StackedDepthSeperableConv1D(16, kernel, e_rates, 1, True),
    BatchNormalization(), Activation(act), UpSampling1D(2),
    
    StackedDepthSeperableConv1D(32, kernel, e_rates, 1, True),
    BatchNormalization(), Activation(act), UpSampling1D(2),

    StackedDepthSeperableConv1D(32, kernel, e_rates, 1, True),
    BatchNormalization(), Activation(act), UpSampling1D(2),
    
    StackedDepthSeperableConv1D(64, kernel, e_rates, 1, False),
    Activation('linear')
])  

## AutoEncoder Wrapper for edf_train
## Tunes for both feature and reconstruction losses
class CustomAutoencoder(Model):
    def __init__(self, encoder, decoder, perceptual_weight=1.0, sd_rate=0.2):
        super(CustomAutoencoder, self).__init__()
        self.spatial_dropout = SpatialDropout1D(sd_rate)
        self.encoder = encoder
        self.decoder = decoder
        self.perceptual_weight = perceptual_weight
        self.mse_loss = MSE()

    def call(self, inputs):
        # Encoding and reconstructing the input
        inputs = self.spatial_dropout(inputs)
        original_features = self.encoder(inputs)
        reconstruction = self.decoder(original_features)
        
        # get features from reconstruction
        reconstructed_features = self.encoder(reconstruction)

        # Compute and add perceptual loss during the call
        perceptual_loss = self.mse_loss(original_features, reconstructed_features)
        self.add_loss(self.perceptual_weight * perceptual_loss)

        # Return only the reconstruction for the main loss computation
        return reconstruction
    
auto_encoder = CustomAutoencoder(encoder, decoder)

class PerceptualClassifier(Model):
    def __init__(self, encoder, decoder, classes, perceptual_weight=0.5, classify_weight=0.5, **kwargs):
        super(PerceptualClassifier, self).__init__(**kwargs)
        
        self.encoder = encoder
        self.decoder = decoder

        channels = self.encoder.input_shape[-1]
        self.expander = create_first_layer(channels)
        self.classifier = create_last_layer(classes)

        self.perceptual_weight = perceptual_weight
        self.classify_weight = classify_weight
        self.percept_loss = lambda y_true, y_pred: 1 + CosineSimilarity(axis=-1)(y_true, y_pred)
        self.cce_loss = CategoricalCrossentropy()
    
    def call(self, inputs):
        expand = self.expander(inputs)
        features = self.encoder(expand)
        output = self.classifier(features)

        reconstruct = self.decoder(features)
        reconstruct_features = self.encoder(reconstruct)

        perceptual_loss = self.perceptual_weight * self.percept_loss(features, reconstruct_features)
        self.add_loss(perceptual_loss)

        return output
    
    def get_loss_function(self):
        return lambda y_true, y_pred: self.classify_weight * self.cce_loss(y_true, y_pred)
    
    def get_lean_model(self):
        inputs = Input(self.expander.input_shape[1:])
        model = Sequential([
            self.expander,
            self.encoder,
            self.classifier
        ])
        outputs = model(inputs)
        return Model(inputs, outputs)

# Custom Activation to maintain zero centered with max standard deviation of 3
def tanh3(x):
    return tf.nn.tanh(x) * 3.5
keras.utils.get_custom_objects().update({'tanh3': tanh3})

## First Layer to convert any channels to 64, standard scaled
def create_first_layer(chs=64):
    return Sequential([
        AddNoiseLayer(0.2),
        Conv1D(chs, 1, padding='same', use_bias=False, activation=tanh3)
    ])

## Last Layer to map latent space to custom classes
def create_last_layer(classes):
    return Sequential([
        Sequential([
            GlobalAveragePooling1D()
        ]),
        Dense(classes, activation='softmax', kernel_regularizer='l2')
    ])
