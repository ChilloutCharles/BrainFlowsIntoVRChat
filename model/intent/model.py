import tensorflow as tf
import numpy as np
import keras

from keras.models import Sequential, Model, clone_model
from keras.layers import Dense, Layer, DepthwiseConv1D, Conv1D
from keras.layers import Activation, Multiply, BatchNormalization, SpatialDropout1D, UpSampling1D, GlobalAveragePooling1D, Input
from keras.losses import MeanSquaredError as MSE, CategoricalCrossentropy
from keras.layers import MultiHeadAttention, LayerNormalization, Reshape

## Spatial Attention (Thanks Summer!)
@keras.utils.register_keras_serializable()
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
    StackedDepthSeperableConv1D(32, kernel, d_rates, 1, True),
    BatchNormalization(), Activation(act), UpSampling1D(2),
    
    StackedDepthSeperableConv1D(32, kernel, d_rates, 1, True),
    BatchNormalization(), Activation(act), UpSampling1D(2),

    StackedDepthSeperableConv1D(32, kernel, d_rates, 1, True),
    BatchNormalization(), Activation(act), UpSampling1D(2),
    
    StackedDepthSeperableConv1D(64, kernel, d_rates, 1, False),
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

## Classifier Model that is guided by pretrained Autoencoder Teacher
class StudentTeacherClassifier(Model):
    def __init__(self, frozen_encoder, frozen_decoder, classes, perceptual_weight=1.0, classify_weight=1.0, **kwargs):
        super(StudentTeacherClassifier, self).__init__(**kwargs)
        
        # create teacher from frozen models
        self.teacher = Sequential([frozen_decoder, frozen_encoder])

        # create student from pieces of unfrozen encoder
        # surround pieces with new first layer and attention layer
        
        first_layer = encoder.layers[:1]
        cloned_encoder = clone_model(frozen_encoder)
        cloned_layers = cloned_encoder.layers[2:]
        for layer in cloned_layers:
            layer.trainable = False

        self.student = Sequential(first_layer + cloned_layers)

        # classifier 
        self.classifier = Sequential([
            GlobalAveragePooling1D(),
            Dense(64, activation='relu'),
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

        # teach the student
        reconstruct_features = self.teacher(features)
        perceptual_loss = self.perceptual_weight * self.percept_loss(features, reconstruct_features)
        self.add_loss(perceptual_loss)

        return output
    
    def get_loss_function(self):
        return lambda y_true, y_pred: self.classify_weight * self.cce_loss(y_true, y_pred)
    
    def build(self, input_shape):
        super(StudentTeacherClassifier, self).build(input_shape)
    
    def get_lean_model(self):
        model = Sequential([
            Input(self.student.input_shape[1:]),
            self.student,
            self.classifier
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

### MASKED AUTOENCODER SECTION ###
# Following along with this paper: Masked Autoencoders Are Scalable Vision Learners
# Using the resulting encoder as a feature extractor that is channel agnostic
# https://arxiv.org/abs/2111.06377

@keras.saving.register_keras_serializable()
class PatchLayer(Layer):
    def __init__(self, patch_shape=(10, 4), **kwargs):
        super(PatchLayer, self).__init__(**kwargs)
        self.patch_shape = patch_shape

    def call(self, inputs):
        # create image-like: (B, 160, 64) -> (B, 160, 64, 1)
        image_like = tf.expand_dims(inputs, -1)
        patch_width = self.patch_shape[0]
        patch_height = self.patch_shape[1]
        
        # Extract patches: shape (B, 16, 16, 40)
        patches = tf.image.extract_patches(
            images=image_like,
            sizes=[1, patch_width, patch_height, 1],
            strides=[1, patch_width, patch_height, 1],
            rates=[1,1,1,1],
            padding='VALID'
        )

        # Flatten each patch: (B, num_patches, patch_dims)
        patches_shape = tf.shape(patches)
        num_patches = patches_shape[-3] * patches_shape[-2]
        patch_dims = patches_shape[-1]
        patches = tf.reshape(patches, [-1, num_patches, patch_dims])
        
        return patches
    
    def build(self, input_shape):
        super(PatchLayer, self).build(input_shape)

@keras.saving.register_keras_serializable()
class Transformer(Layer):
    def __init__(self, ffn_dim, out_dim, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.attn = MultiHeadAttention(4, 16)
        self.ffn = Sequential([
            Dense(ffn_dim, activation='gelu'),
            Dense(out_dim, activation='gelu')
        ])
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
    
    def build(self, input_shape):
        super(Transformer, self).build(input_shape)
    
    def call(self, inputs):
        attn_out = self.attn(inputs, inputs)
        attn_out = self.ln1(attn_out + inputs)
        ffn_out = self.ffn(attn_out) 
        ffn_out = self.ln2(ffn_out + attn_out)
        return ffn_out

@keras.utils.register_keras_serializable()
class TrainablePositionalEmbedding(Layer):
    def __init__(self, max_len=160, embed_dim=64, **kwargs):
        super(TrainablePositionalEmbedding, self).__init__(**kwargs)
        self.max_len = max_len  
        self.embed_dim = embed_dim 
        self.tanh = Activation('tanh') # Smooth Gradient please

    def build(self, input_shape):
        # Initialize a trainable positional embedding of shape (max_len, embed_dim)
        self.positional_embeddings = self.add_weight(
            name='pos_embedding',
            shape=(self.max_len, self.embed_dim),
            initializer='random_normal',
            trainable=True
        )
        super(TrainablePositionalEmbedding, self).build(input_shape)

    def call(self, inputs):
        # Add the positional embeddings to the input tensor
        seq_len = tf.shape(inputs)[1]  # Get the sequence length (number of patches)
        pos_embed = self.tanh(self.positional_embeddings[:seq_len, :]) # smoothed pos_embed
        return inputs + pos_embed
    
@keras.saving.register_keras_serializable()
class SinusoidPositionalEmbedding(Layer):
    def __init__(self, max_len=160, embed_dim=64, **kwargs):
        super(SinusoidPositionalEmbedding, self).__init__(**kwargs)
        self.max_len = max_len  
        self.embed_dim = embed_dim  

        # Generate the positional encoding matrix for all positions and embedding dimensions
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))
        
        pe = np.zeros((self.max_len, self.embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = np.cos(position * div_term)  # Apply cos to odd indices
        
        # Convert to TensorFlow tensor and add a batch dimension
        self.positional_embeddings = tf.constant(pe, dtype=tf.float32)

    def build(self, input_shape):
        super(SinusoidPositionalEmbedding, self).build(input_shape)

    def call(self, inputs):
        # Add the positional encodings to the input tensor
        seq_len = tf.shape(inputs)[1]
        return inputs + self.positional_embeddings[:seq_len, :]

@keras.saving.register_keras_serializable()
class MaskedAutoEncoder(Model):
    def __init__(self, times=160, ffn_dim=32, emb_dim=64, out_dim=64, patch_shape=(10, 4), **kwargs):
        super(MaskedAutoEncoder, self).__init__(**kwargs)

        patch_dim = patch_shape[0] * patch_shape[1]
        patch_count_w = times//patch_shape[0] 
        patch_count_h = emb_dim//patch_shape[1]
        patch_count = patch_count_w * patch_count_h

        self.patcher = Sequential([
            Input((None, None)),
            PatchLayer(patch_shape),
            Dense(emb_dim)
        ], name='patcher')
        
        self.encoder_position = TrainablePositionalEmbedding(patch_count, emb_dim)
        self.decoder_position = SinusoidPositionalEmbedding(patch_count, emb_dim)

        self.encoder = Sequential([Input((None, emb_dim))] +  [Transformer(ffn_dim, emb_dim) for _ in range(8)], name='encoder')
        self.decoder = Sequential([Input((None, emb_dim))] +  [Transformer(ffn_dim, emb_dim) for _ in range(2)], name='decoder')

        self.unpatch_recover = Sequential([
            Dense(patch_dim),
            Reshape((times, out_dim))
        ], name='unpatch_recover')

        self.mask_token = tf.Variable(tf.random.normal((1, patch_count, 1)), trainable=True)
        self.num_mask = int(0.75 * patch_count)

    def call(self, inputs):
        # patch, linearly embed, then patch position embed
        embedding = self.patcher(inputs)
        embedding = self.encoder_position(embedding)

        # get embed shape for later use
        embed_shape = tf.shape(embedding)

        # create unmasked indices and gather unmasked patches
        rand_indices = tf.argsort(tf.random.uniform(shape=embed_shape[:2]), axis=-1)
        unmask_indices = rand_indices[:, self.num_mask :]
        unmasked_embeds = tf.gather(embedding, unmask_indices, axis=1, batch_dims=1)

        # send unmasked patches through encoder
        features = self.encoder(unmasked_embeds)

        # reintroduce masked portions with the mask token, apply positional embed too
        decoder_input = tf.broadcast_to(self.mask_token, embed_shape)
        decoder_input = self.hard_ass_scatter_update(decoder_input, unmask_indices, features)
        decoder_input = self.decoder_position(decoder_input)
        
        # send mask tokenized full data to decoder and reconstruct original patches
        reconstruct_embedding = self.decoder(decoder_input)

        # undo linear embedding and stich patches back together
        reconstruct = self.unpatch_recover(reconstruct_embedding)

        return reconstruct
    
    def hard_ass_scatter_update(self, tensor, indices, updates):
        batch_size = tf.shape(tensor)[0]
        num_updates = tf.shape(indices)[1]
        embed_dim = tf.shape(tensor)[2]
        
        # Create batch indices (for scatter updates) with shape (batch_size * num_updates, 1)
        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, num_updates])
        batch_indices = tf.reshape(batch_indices, (-1, 1))  # Shape: (batch_size * num_updates, 1)

        indices_reshaped = tf.reshape(indices, (-1, 1))  # Shape: (batch_size * num_updates, 1)
        scatter_indices = tf.concat([batch_indices, indices_reshaped], axis=1)  # Shape: (batch_size * num_updates, 2)
        updates_reshaped = tf.reshape(updates, (-1, embed_dim))  # Shape: (batch_size * num_updates, embed_dim)

        updated_tensor = tf.tensor_scatter_nd_update(tensor, scatter_indices, updates_reshaped)

        return updated_tensor
    
    def assemble_feature_extractor(self):
        return Sequential([
            self.patcher,
            self.encoder_position,
            self.encoder
        ])
    
def create_classifier(feature_extractor, classes):
    [patcher, position, encoder] = feature_extractor.layers
    patcher.trainable = False

    # Set positonal embedding and half of last layer of encoder to trainable
    position.trainable = True
    for layer in encoder.layers[:-1]:
        layer.trainable = False
    last_tfm = encoder.layers[-1]
    last_tfm.attn.trainable = False
    last_tfm.ln1.trainable = False

    return Sequential([
        BatchNormalization(), # input data was normalized during pretraining. Equivalent!
        feature_extractor,
        GlobalAveragePooling1D(),
        Dense(classes, activation='softmax')
    ], name='classifier')