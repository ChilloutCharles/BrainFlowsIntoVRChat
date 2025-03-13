import tensorflow as tf
import numpy as np
import keras

from keras.models import Sequential, Model, clone_model
from keras.layers import Dense, Layer, DepthwiseConv1D, Conv1D, MaxPooling2D
from keras.layers import Activation, Multiply, BatchNormalization, SpatialDropout1D, UpSampling1D, GlobalAveragePooling1D, Input
from keras.layers import MultiHeadAttention, LayerNormalization, Reshape
from keras.losses import MeanSquaredError as MSE, CategoricalCrossentropy

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
# input has been preprocessed with notch and bandpass filtering
# then turned into a (160, 64, 3) image using multi resolution analysis
# https://pywavelets.readthedocs.io/en/latest/ref/mra.html

@keras.saving.register_keras_serializable()
class PatchLayer(Layer):
    def __init__(self, patch_shape=(10, 4), **kwargs):
        super(PatchLayer, self).__init__(**kwargs)
        self.patch_shape = patch_shape

    def call(self, inputs):
        patch_width = self.patch_shape[0]
        patch_height = self.patch_shape[1]
        
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, patch_width, patch_height, 1],
            strides=[1, patch_width, patch_height, 1],
            rates=[1,1,1,1],
            padding='VALID'
        )

        patches_shape = tf.shape(patches)
        num_patches = patches_shape[-3] * patches_shape[-2]
        patch_dims = patches_shape[-1]
        patches = tf.reshape(patches, [-1, num_patches, patch_dims])
        
        return patches
    
    def build(self, input_shape):
        super(PatchLayer, self).build(input_shape)

@keras.saving.register_keras_serializable()
class MultiHeadSelfAttention(Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.attn = MultiHeadAttention(num_heads, key_dim)
    
    def call(self, inputs):
        return self.attn(inputs, inputs)
    
    def build(self, input_shape):
        super(MultiHeadSelfAttention, self).build(input_shape)

@keras.saving.register_keras_serializable()
class Transformer(Layer):
    def __init__(self, num_head, ffn_dim, out_dim, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        key_dim = out_dim//num_head
        self.attn = MultiHeadSelfAttention(num_head, key_dim)
        self.ffn = Sequential([
            Dense(ffn_dim, activation='gelu'),
            Dense(out_dim, activation='linear')
        ])
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
    
    def build(self, input_shape):
        super(Transformer, self).build(input_shape)
    
    def call(self, inputs):
        attn_out = self.ln1(inputs)
        attn_out = self.attn(attn_out) + inputs
        ffn_out = self.ln2(attn_out) 
        ffn_out = self.ffn(ffn_out) + attn_out
        return ffn_out

@keras.utils.register_keras_serializable()
class TrainablePositionalEmbedding(Layer):
    def __init__(self, max_len=160, **kwargs):
        super(TrainablePositionalEmbedding, self).__init__(**kwargs)
        self.max_len = max_len
        self.tanh = Activation('tanh')

    def build(self, input_shape):
        # Define a trainable positional embedding unique to the sequence dimension
        self.positional_embeddings = self.add_weight(
            name='positional_embeddings',
            shape=(self.max_len, 1),  # One unique value per time step
            initializer='random_normal',
            trainable=True,
            regularizer='l2'
        )
        super(TrainablePositionalEmbedding, self).build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos_embeddings = tf.tile(self.positional_embeddings[:seq_len, :], [1, tf.shape(inputs)[-1]])
        pos_embeddings = self.tanh(pos_embeddings)
        return inputs + pos_embeddings
    
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
    def __init__(self, input_shape, patch_shape, mask_ratio=0.8, num_heads=5, ae_size=(10, 1), loss_func=None, loss_p=0.9, **kwargs):
        super(MaskedAutoEncoder, self).__init__(**kwargs)
        self.input_shape = input_shape

        patch_dim = patch_shape[0] * patch_shape[1] * input_shape[2]
        patch_count_h = input_shape[0]//patch_shape[0]
        patch_count_w = input_shape[1]//patch_shape[1]
        patch_count = patch_count_h * patch_count_w

        embed_dim = patch_dim * 2
        ffn_dim = embed_dim * 4

        self.patch_position = SinusoidPositionalEmbedding(patch_count, embed_dim)

        self.patcher = Sequential([
            Input((None, None, input_shape[2])),
            PatchLayer(patch_shape)
        ], name='patcher')

        self.project = Sequential([
            Input((None, patch_dim)),
            Dense(embed_dim, use_bias=False),
        ], name='project')

        self.encoder = Sequential([Input((None, embed_dim))] +  [Transformer(num_heads, ffn_dim, embed_dim) for _ in range(ae_size[0])], name='encoder')
        self.decoder = Sequential([Input((None, embed_dim))] +  [Transformer(num_heads, ffn_dim, embed_dim) for _ in range(ae_size[1])], name='decoder')

        self.unproject = Dense(patch_dim, use_bias=False)
        self.recover = Reshape(self.input_shape)
        
        self.mask_token = tf.Variable(tf.random.normal((1, patch_count, 1)), trainable=True)
        self.num_mask = int(mask_ratio * patch_count)

        # Internal Loss 
        self.loss_func = loss_func
        self.loss_p = loss_p

    def call(self, inputs):
        # patch, linearly project, and apply static position embedding
        input_patches = self.patcher(inputs)
        embedding = self.project(input_patches)
        embedding = self.patch_position(embedding)
        
        # get embedding shape for later use
        embed_shape = tf.shape(embedding)

        # create indices and gather unmasked patches
        rand_indices = tf.argsort(tf.random.uniform(shape=embed_shape[:2]), axis=-1)
        mask_indices = rand_indices[:, :self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        unmasked_embeds = tf.gather(embedding, unmask_indices, axis=1, batch_dims=1)

        # send unmasked patches through encoder
        features = self.encoder(unmasked_embeds)

        # reintroduce masked portions with the mask token and apply positional embed
        decoder_input = tf.broadcast_to(self.mask_token, embed_shape)
        decoder_input = self.hard_ass_scatter_update(decoder_input, unmask_indices, features)
        decoder_input = self.patch_position(decoder_input)
        
        # send mask tokenized full patch sequence to decoder
        reconstruct_embedding = self.decoder(decoder_input)

        # project back to patch dims
        reconstruct_patches = self.unproject(reconstruct_embedding)

        # do patches loss if a loss function was set
        if self.loss_func:
            masked_reconstruct = tf.gather(reconstruct_patches, mask_indices, axis=1, batch_dims=1)
            masked_input = tf.gather(input_patches, mask_indices, axis=1, batch_dims=1)
            mask_loss = self.loss_func(masked_input, masked_reconstruct)

            unmasked_reconstruct = tf.gather(reconstruct_patches, unmask_indices, axis=1, batch_dims=1)
            unmasked_input = tf.gather(input_patches, unmask_indices, axis=1, batch_dims=1)
            unmask_loss = self.loss_func(unmasked_input, unmasked_reconstruct)

            loss = self.loss_p * mask_loss + (1.0 - self.loss_p) * unmask_loss
            self.add_loss(loss)
        
        # stich patches back together
        reconstruct = self.recover(reconstruct_patches)
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
        feature_extractor = Sequential([
            self.patcher,
            self.project,
            self.patch_position,
            self.encoder
        ])
        feature_extractor.build(input_shape=(None, *self.input_shape))
        return feature_extractor

    def build(self, input_shape):
        super(MaskedAutoEncoder, self).build(input_shape)

@keras.utils.register_keras_serializable()
class ShuffleLayer(Layer):
    def __init__(self, **kwargs):
        super(ShuffleLayer, self).__init__(**kwargs)
    def call(self, inputs, training=None):
        if training:
            rand_indices = tf.argsort(tf.random.uniform(shape=tf.shape(inputs)[:2]), axis=-1)
            inputs = tf.gather(inputs, rand_indices, axis=1, batch_dims=1)
        return inputs
    def build(self, input_shape):
        super(ShuffleLayer, self).build(input_shape)

def create_classifier(feature_extractor, classes, input_shape):
    [
        patcher,
        project,
        _,
        encoder
    ] = feature_extractor.layers

    # create a pooling layer to map input channels to what the patcher can handle
    chans = input_shape[1]
    pool_size = (1, 1)
    if chans % 4 != 0:
        new_chans = (chans // 4) * 4
        pool_size = (1, max(2, chans - new_chans))
    patch_matcher = MaxPooling2D(pool_size=pool_size)
    
    # freeze patcher and projection layers
    patcher.trainable = False
    project.trainable = False

    # freeze all layers of encoder
    # except first transform layer norms and last half of the last transform
    for layer in encoder.layers[1:-1]:
        layer.trainable = False
    
    first_tfm = encoder.layers[0]
    first_tfm.attn.trainable = False
    first_tfm.ffn.trainable = False

    last_tfm = encoder.layers[-1]
    last_tfm.ln1.trainable = False
    last_tfm.attn.trainable = False

    # replace positional embedder with user custom
    patch_position = TrainablePositionalEmbedding()

    # create self attention pooling
    embed_dim = encoder.input_shape[-1]
    num_heads = 8
    pool = Sequential([
        LayerNormalization(),
        MultiHeadSelfAttention(num_heads, embed_dim//num_heads),
        GlobalAveragePooling1D(),
    ], name='GlobalSelfAttentionPooling1D')

    return Sequential([
        patch_matcher,
        patcher,
        project,
        patch_position,
        ShuffleLayer(),
        encoder,
        pool,
        Dense(classes, activation='softmax')
    ], name='classifier')

