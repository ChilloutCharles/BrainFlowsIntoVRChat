import tensorflow as tf
import keras

from keras.models import Sequential, Model, clone_model
from keras.layers import Dense, Layer, DepthwiseConv1D, Conv1D, Attention
from keras.layers import Activation, Multiply, BatchNormalization, SpatialDropout1D, UpSampling1D, GlobalAveragePooling1D, Input, Dropout
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

kernel = 3
e_rates = [1, 2, 4]
d_rates = list(reversed(e_rates))
act = 'elu'

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
    
    StackedDepthSeperableConv1D(32, kernel, e_rates, 2, True),
    BatchNormalization(), Activation(act), # (40, 32)
    
    StackedDepthSeperableConv1D(32, kernel, e_rates, 2, True),
    BatchNormalization(), Activation(act), # (20, 32)

    StackedDepthSeperableConv1D(32, kernel, e_rates, 1, False), 
    Activation('linear')
])

decoder = Sequential([
    StackedDepthSeperableConv1D(32, kernel, e_rates, 1, True),
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

## Classifier Model that trains for both classification and perceptual targets
class PerceptualClassifier(Model):
    def __init__(self, frozen_encoder, frozen_decoder, classes, perceptual_weight=1.0, classify_weight=1.0, **kwargs):
        super(PerceptualClassifier, self).__init__(**kwargs)
        
        # create teacher from frozen models
        self.teacher = Sequential([frozen_decoder, frozen_encoder])

        # create student from pieces of unfrozen encoder
        first_layer = [StackedDepthSeperableConv1D(64, kernel, e_rates, 2, True)]
        cloned_encoder = clone_model(frozen_encoder)
        cloned_encoder.set_weights(cloned_encoder.get_weights())
        encoder_layers = first_layer + cloned_encoder.layers[1:]
        self.student = Sequential(encoder_layers)

        # classifier 
        self.classifier = Sequential([
            GlobalAveragePooling1D(),
            Dense(classes, activation='softmax', kernel_regularizer='l2')
        ])

        # perceptual and classification losses
        self.perceptual_weight = perceptual_weight
        self.classify_weight = classify_weight
        self.percept_loss = MSE()
        self.cce_loss = CategoricalCrossentropy()
    
    def call(self, inputs):
        # predict class
        features = self.student(inputs)
        output = self.classifier(features)

        # teach 
        reconstruct_features = self.teacher(features)
        perceptual_loss = self.perceptual_weight * self.percept_loss(features, reconstruct_features)
        self.add_loss(perceptual_loss)

        return output
    
    def get_loss_function(self):
        return lambda y_true, y_pred: self.classify_weight * self.cce_loss(y_true, y_pred)
    
    def get_lean_model(self):
        model = Sequential([
            Input(self.student.input_shape[1:]),
            self.student,
            self.classifier
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model
