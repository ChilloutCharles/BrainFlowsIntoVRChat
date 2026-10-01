import tensorflow as tf
import numpy as np
import keras

from keras.models import Sequential, Model, clone_model
from keras.layers import Dense, Layer, DepthwiseConv1D, Conv1D, MaxPooling2D, SeparableConv1D, GaussianNoise 
from keras.layers import Activation, Multiply, BatchNormalization, SpatialDropout1D, UpSampling1D, GlobalAveragePooling1D, Input, Dropout
from keras.layers import MultiHeadAttention, LayerNormalization, Reshape, Flatten
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
class RotaryPositionalEmbedding(Layer):
    def __init__(self, max_len=160, head_dim=12, num_registers=4, theta=10000.0, **kwargs):
        super(RotaryPositionalEmbedding, self).__init__(**kwargs)
        self.max_len = max_len
        self.head_dim = head_dim
        self.theta = theta
        self.num_registers = num_registers

        # 1. Compute inverse frequencies for half the head dimension space
        half_dim = self.head_dim // 2
        div_term = 1.0 / (self.theta ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
        
        # 2. Shift the position index generation (registers remain at 0)
        position = np.zeros(self.max_len, dtype=np.float32)
        if self.max_len > self.num_registers:
            position[self.num_registers:] = np.arange(1, self.max_len - self.num_registers + 1, dtype=np.float32)
        
        position = position[:, np.newaxis]
        angles = position * div_term

        # 3. Duplicate angles across the full head_dim
        full_angles = np.concatenate([angles, angles], axis=-1)

        # 4. Compute cos and sin, then reshape for MHA broadcasting: (1, max_len, 1, head_dim)
        cos_pe = np.cos(full_angles)[np.newaxis, :, np.newaxis, :]
        sin_pe = np.sin(full_angles)[np.newaxis, :, np.newaxis, :]

        # 5. Lock them into static constants to avoid graph isolation errors
        self.cos_cached = tf.constant(cos_pe, dtype=tf.float32)
        self.sin_cached = tf.constant(sin_pe, dtype=tf.float32)

    def build(self, input_shape):
        super(RotaryPositionalEmbedding, self).build(input_shape)

    def _rotate_half(self, x):
        half_dim = self.head_dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return tf.concat([-x2, x1], axis=-1)

    def call(self, x):
        # x shape from your trace: (batch, seq_len, num_heads, head_dim)
        # e.g., (None, None, 6, 40)
        seq_len = tf.shape(x)[1]
        
        # Slice the cached embedding to match the current dynamic sequence length
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        
        # Apply the standard RoPE formulation: R(x) = x * cos + rotate_half(x) * sin
        return (x * cos) + (self._rotate_half(x) * sin)

    def get_config(self):
        config = super(RotaryPositionalEmbedding, self).get_config()
        config.update({
            "max_len": self.max_len,
            "head_dim": self.head_dim,
            "theta": self.theta,
            "num_registers": self.num_registers,
        })
        return config


@keras.saving.register_keras_serializable()
class RoPEMultiHeadAttention(MultiHeadAttention):
    def __init__(self, num_heads, key_dim, register_count, **kwargs):
        super().__init__(num_heads=num_heads, key_dim=key_dim, **kwargs)
        self.rope_layer = RotaryPositionalEmbedding(160*5, key_dim, register_count)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        # Intercept: Keras passes these tensors to us post-projection with shape (B, S, H, D)
        query = self.rope_layer(query)
        key = self.rope_layer(key)
        
        # Hand the rotated tensors back to Keras's underlying attention math engine
        return super()._compute_attention(
            query, key, value, attention_mask=attention_mask, training=training
        )

@keras.saving.register_keras_serializable()
class MultiHeadSelfAttention(Layer):
    def __init__(self, num_heads, key_dim, register_count, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.attn = RoPEMultiHeadAttention(num_heads, key_dim, register_count)
    
    def call(self, inputs):
        return self.attn(query=inputs, key=inputs, value=inputs)
    
    def build(self, input_shape):
        super(MultiHeadSelfAttention, self).build(input_shape)

@keras.saving.register_keras_serializable()
class Transformer(Layer):
    def __init__(self, num_head, ffn_dim, out_dim, register_count, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        key_dim = out_dim//num_head
        self.attn = MultiHeadSelfAttention(num_head, key_dim, register_count)
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
class ExtractRegisterLayer(Layer):
    def __init__(self, register_count=5, **kwargs):
        super(ExtractRegisterLayer, self).__init__(**kwargs)
        self.register_count=register_count
    def call(self, inputs):
        return inputs[:, 0:self.register_count, :]
    def build(self, input_shape):
        super(ExtractRegisterLayer, self).build(input_shape)

@keras.utils.register_keras_serializable()
class RemoveRegisterLayer(Layer):
    def __init__(self, register_count=5, **kwargs):
        super(RemoveRegisterLayer, self).__init__(**kwargs)
        self.register_count = register_count
    def call(self, inputs):
        return inputs[:, self.register_count:, :]
    def build(self, input_shape):
        super(RemoveRegisterLayer, self).build(input_shape)
    def get_config(self):
        config = super(RemoveRegisterLayer, self).get_config()
        config.update({
            "register_count": self.register_count
        })
        return config

@keras.utils.register_keras_serializable()
class PrependRegisterLayer(Layer):
    def __init__(self, register_count=5, **kwargs):
        super(PrependRegisterLayer, self).__init__(**kwargs)
        self.register_count = register_count

    def build(self, input_shape):
        self.register_tokens = self.add_weight(
            name="register_tokens",
            shape=(1, self.register_count, input_shape[-1]),
            initializer="random_normal",
            trainable=True
        )
        super(PrependRegisterLayer, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        register_tokens_batched = tf.tile(self.register_tokens, [batch_size, 1, 1])
        return tf.concat([register_tokens_batched, inputs], axis=1)

    def get_config(self):
        config = super(PrependRegisterLayer, self).get_config()
        config.update({
            "register_count": self.register_count
        })
        return config

@keras.saving.register_keras_serializable()
class MaskedAutoEncoder(Model):
    def __init__(self, input_shape, patch_shape, mask_ratio=0.8, num_heads=5, ae_size=(10, 1), loss_func=None, loss_p=0.9, **kwargs):
        super(MaskedAutoEncoder, self).__init__(**kwargs)
        self.input_shape = input_shape

        patch_dim = patch_shape[0] * patch_shape[1] * input_shape[2]
        patch_count_h = input_shape[0]//patch_shape[0]
        patch_count_w = input_shape[1]//patch_shape[1]
        patch_count = patch_count_h * patch_count_w

        encoder_embed_dim = patch_dim * 2
        ffn_dim = encoder_embed_dim * 4
        decoder_embed_dim = encoder_embed_dim // 2


        self.patcher = Sequential([
            Input((None, None, input_shape[2])),
            PatchLayer(patch_shape)
        ], name='patcher')

        self.project = Sequential([
            Input((None, patch_dim)),
            Dense(encoder_embed_dim, use_bias=False),
        ], name='project')

        register_count = 5

        self.encoder = Sequential([Input((None, encoder_embed_dim))] +  [Transformer(num_heads, ffn_dim, encoder_embed_dim, register_count) for _ in range(ae_size[0])], name='encoder')

        self.decoder_project = Dense(decoder_embed_dim, use_bias=False, name='decoder_project')

        self.decoder = Sequential([Input((None, decoder_embed_dim))] +  [Transformer(num_heads, ffn_dim, decoder_embed_dim, register_count) for _ in range(ae_size[1])], name='decoder')

        self.unproject = Dense(patch_dim, use_bias=False)
        self.recover = Reshape(self.input_shape)
        
        self.mask_token = tf.Variable(tf.random.normal((1, patch_count, 1)), trainable=True)
        self.num_mask = int(mask_ratio * patch_count)

        self.encoder_regadd = PrependRegisterLayer(register_count)
        self.encoder_regrem = RemoveRegisterLayer(register_count)

        self.decoder_regadd = PrependRegisterLayer(register_count)
        self.decoder_regrem = RemoveRegisterLayer(register_count)

        # Internal Loss 
        self.loss_func = loss_func
        self.loss_p = loss_p

    def call(self, inputs):
        # patch, linearly project, and apply static position embedding
        input_patches = self.patcher(inputs)
        embedding = self.project(input_patches)
        
        # get embedding shape for later use
        embed_shape = tf.shape(embedding)

        # create indices and gather unmasked patches
        rand_indices = tf.argsort(tf.random.uniform(shape=embed_shape[:2]), axis=-1)
        mask_indices = rand_indices[:, :self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        unmasked_embeds = tf.gather(embedding, unmask_indices, axis=1, batch_dims=1)

        # prepend encoder register tokens
        unmasked_embeds = self.encoder_regadd(unmasked_embeds)

        # send unmasked patches through encoder
        features = self.encoder(unmasked_embeds)

        # separate register tokens from the features
        features = self.encoder_regrem(features)

        # reintroduce masked portions with the mask token, apply positional encode, then scatter the features
        decoder_input = tf.broadcast_to(self.mask_token, embed_shape)
        decoder_input = self.hard_ass_scatter_update(decoder_input, unmask_indices, features)

        # project into decoder embed dim
        decoder_input = self.decoder_project(decoder_input)

        # prepend decoder register tokens
        decoder_input = self.decoder_regadd(decoder_input)
        
        # send mask tokenized full patch sequence to decoder
        reconstruct_embedding = self.decoder(decoder_input)

        # slice off register tokens from reconstruction
        reconstruct_embedding = self.decoder_regrem(reconstruct_embedding)

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
            self.encoder_regadd,
            self.encoder,
            self.decoder_project
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
        encoder_regadd,
        encoder,
        decoder_project
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

    # CLS finetuning
    encoder_regadd.trainable = False
    for layer in encoder.layers[:-2]:
        layer.trainable = False
    
    # Freeze Decoder Projection
    decoder_project.trainable = False

    return Sequential([
        patch_matcher,
        patcher,
        project,
        encoder_regadd,
        encoder,
        ExtractRegisterLayer(1),
        decoder_project,
        Flatten(),
        Dense(classes, activation='softmax')
    ], name='classifier')

