'''
This file contains an implementation of ABAE (details in another comment below)
and a series of models that build on the basic structure of ABAE.
'''

from tensorflow.python.keras.layers.dense_attention import Attention
from models.base_models import UnsupervisedDeepNeuralNetworkModel
from model_helpers.custom_layers import *

from tensorflow.keras.layers import Input, Dense, Activation, Concatenate, Attention, LSTM, GlobalAveragePooling1D, Embedding
from tensorflow.keras.layers import Average as k_Average
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Constant
from tensorflow import eye as tf_eye
import numpy as np


class SimpleABAE(UnsupervisedDeepNeuralNetworkModel):
    ''' 
    Implementation of the Attention-Based Aspect Extraction model from
    He et al.'s "An Unsupervised Neural Attention Model for Aspect Extraction"

    This version of ABAE is slightly simplified since the positive and negative
    inputs have the same shape.

    Code is based on:
    - https://github.com/ruidan/Unsupervised-Aspect-Extraction/blob/master/code/
    - https://github.com/luckmoon/Unsupervised-Aspect-Extraction/blob/master/code/
    '''
    def __init__(self, neg_size, win_size, emb_dim, output_size, optimizer='adam', loss='categorical_crossentropy', loss_weights=None):
        self._neg_size = neg_size

        super().__init__(win_size, emb_dim, output_size, optimizer=optimizer, loss=loss, loss_weights=loss_weights)
    
    def _ortho_reg(self, weight_matrix, ortho_reg=0.1):
    # orthogonal regularization for aspect embedding matrix
        w_n = weight_matrix / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(weight_matrix), axis=-1, keepdims=True)), K.floatx())
        reg = K.sum(K.square(K.dot(w_n, K.transpose(w_n)) - tf_eye(w_n.shape[0])))

        return ortho_reg * reg

    @staticmethod
    def max_margin_loss(y_true, y_pred):
        return K.mean(y_pred)

    @staticmethod
    def batch_generator(data, batch_size, neg_size, shuffle=True):
        n_batch = data.shape[0] / batch_size
        batch_count = 0
        if shuffle: np.random.shuffle(data)
        
        while True:
            pos_batch = data[batch_count * batch_size: (batch_count+1) * batch_size]
            
            neg_batch_inds = np.random.choice(data.shape[0], size=pos_batch.shape[0], replace=False)
            neg_batch = data[neg_batch_inds]

            batch_count += 1

            yield([pos_batch, neg_batch], np.ones((pos_batch.shape[0], 1)))

    def _compile_model(self):
        maxlen = self._n
        aspect_size = self._output_size

        # Inputs
        emb_input = Input(shape=(maxlen, self._emb_dim), dtype='float32', name='sentence_input')
        neg_input = Input(shape=(maxlen, self._emb_dim), dtype='float32', name='neg_input')

        # Compute sentence representation
        # e_w = word_emb(sentence_input)
        y_s = Average(name='sentence_avg')(emb_input)
        att_weights = Attention_ABAE(name='att')([emb_input, y_s])
        z_s = WeightedSum()([emb_input, att_weights])

        # Compute representations of negative instances
        # e_neg = word_emb(neg_input)
        z_n = Average()(neg_input)

        # Reconstruction
        p_t = Dense(aspect_size)(z_s)
        p_t = Activation('softmax', name='p_t')(p_t)
        r_s = WeightedAspectEmb(aspect_size, self._emb_dim, name='aspect_emb',
                                W_regularizer=self._ortho_reg)(p_t)

        # Loss
        loss = MaxMargin(neg_size=self._neg_size, name='max_margin')([z_s, z_n, r_s])
        model = Model(inputs=[emb_input, neg_input], outputs=loss)
        model.compile(optimizer=self._optimizer, loss=SimpleABAE.max_margin_loss, metrics=[SimpleABAE.max_margin_loss])

        return model
    
    @classmethod
    def load_model(cls, model_path):
        custom_layers = {
            'Average': Average,
            'Attention_ABAE': Attention_ABAE,
            'WeightedSum': WeightedSum,
            'WeightedAspectEmb': WeightedAspectEmb,
            'MaxMargin': MaxMargin,
            'max_margin_loss': cls.max_margin_loss
        }

        return load_model(model_path, custom_objects=custom_layers)

    def get_aspect_embedding_weights(self):
        return self._model.get_layer('aspect_emb').get_weights()[0]

    def get_softmax_layer(self):
        return self._model.get_layer('p_t').output

    def get_input_layers(self):
        return self._model.inputs

    def build_softmax_model(self):
        '''
        This method creates a new model based on a _trained_ ABAE model's layers.
        The new model stops at the ABAE model's softmax layer. Running new samples
        through this model exposes the probability mass ABAE assigns for each new sample
        across each aspect.
        '''

        input_layers = self.get_input_layers()
        softmax_layer = self.get_softmax_layer()

        softmax_model = Model(input_layers, softmax_layer)

        return softmax_model
    
class SimpleABAE_Emb(SimpleABAE):
    def __init__(
        self,
        num_tokens,
        emb_matrix,
        neg_size, 
        win_size, 
        emb_dim, 
        output_size, 
        optimizer='adam', 
        loss='categorical_crossentropy', 
        loss_weights=None,
        trainable_emb_layer=False
    ):
        self._embedding_layer = self._create_embedding_layer(num_tokens, emb_dim, emb_matrix, trainable_emb_layer)
        super().__init__(neg_size, win_size, emb_dim, output_size, optimizer=optimizer, loss=loss, loss_weights=loss_weights)
        
    def _create_embedding_layer(self, num_tokens, emb_dim, emb_matrix, trainable_emb_layer):
        return Embedding(
            num_tokens,
            emb_dim,
            embeddings_initializer=Constant(emb_matrix),
            trainable=trainable_emb_layer
        )
    
    def _compile_model(self):
        maxlen = self._n
        aspect_size = self._output_size
        
        # Embedding layers
        embedding_layer = self._embedding_layer
        
        # Inputs
        emb_input = Input(shape=(maxlen,), dtype='float32', name='sentence_input')
        neg_input = Input(shape=(maxlen,), dtype='float32', name='neg_input')

        # Compute sentence representation
        pos_emb = embedding_layer(emb_input)
        y_s = Average(name='sentence_avg')(pos_emb)
        att_weights = Attention_ABAE(name='att')([pos_emb, y_s])
        z_s = WeightedSum()([pos_emb, att_weights])

        # Compute representations of negative instances
        neg_emb = embedding_layer(neg_input)
        z_n = Average()(neg_emb)

        # Reconstruction
        p_t = Dense(aspect_size)(z_s)
        p_t = Activation('softmax', name='p_t')(p_t)
        r_s = WeightedAspectEmb(aspect_size, self._emb_dim, name='aspect_emb',
                                W_regularizer=self._ortho_reg)(p_t)

        # Loss
        loss = MaxMargin(neg_size=self._neg_size, name='max_margin')([z_s, z_n, r_s])
        model = Model(inputs=[emb_input, neg_input], outputs=loss)
        model.compile(optimizer=self._optimizer, loss=SimpleABAE.max_margin_loss, metrics=[SimpleABAE.max_margin_loss])

        return model

class New_ABAE(SimpleABAE):
    ''' This class mimics the ABAE architecture, but uses a standard Keras Attention layer. '''
    def _compile_model(self):
        maxlen = self._n
        aspect_size = self._output_size

        # Inputs
        emb_input = Input(shape=(maxlen, self._emb_dim), dtype='float32', name='sentence_input')
        neg_input = Input(shape=(maxlen, self._emb_dim), dtype='float32', name='neg_input')

        # Compute sentence representation
        y_s = Average(name='sentence_avg')(emb_input)
        att_weights = Attention(name='att')([emb_input, y_s])
        pooled_weights = GlobalAveragePooling1D(name='pool', data_format='channels_first')(att_weights) #TODO: investigate max pooling
        z_s = WeightedSum()([emb_input, pooled_weights])

        # Compute representations of negative instances
        # e_neg = word_emb(neg_input)
        z_n = Average()(neg_input)

        # Reconstruction
        p_t = Dense(aspect_size)(z_s)
        p_t = Activation('softmax', name='p_t')(p_t)
        r_s = WeightedAspectEmb(aspect_size, self._emb_dim, name='aspect_emb',
                                W_regularizer=self._ortho_reg)(p_t)

        # Loss
        loss = MaxMargin(neg_size=self._neg_size, name='max_margin')([z_s, z_n, r_s])
        model = Model(inputs=[emb_input, neg_input], outputs=loss)
        model.compile(optimizer=self._optimizer, loss=SimpleABAE.max_margin_loss, metrics=[SimpleABAE.max_margin_loss])

        return model
    
class New_ABAE_Emb(SimpleABAE_Emb):
    ''' This class uses a Keras Embedding layer, but otherwise is identical to New_ABAE. '''
    def _compile_model(self):
        maxlen = self._n
        aspect_size = self._output_size

        # Embedding layers
        embedding_layer = self._embedding_layer

        # Inputs
        emb_input = Input(shape=(maxlen,), dtype='float32', name='sentence_input')
        neg_input = Input(shape=(maxlen,), dtype='float32', name='neg_input')

        # Compute sentence representation
        pos_emb = embedding_layer(emb_input)
        y_s = Average(name='sentence_avg')(pos_emb)
        att_weights = Attention(name='att')([pos_emb, y_s])
        pooled_weights = GlobalAveragePooling1D(name='pool', data_format='channels_first')(att_weights) #TODO: investigate max pooling
        z_s = WeightedSum()([pos_emb, pooled_weights])

        # Compute representations of negative instances
        neg_emb = embedding_layer(neg_input)
        z_n = Average()(neg_emb)

        # Reconstruction
        p_t = Dense(aspect_size)(z_s)
        p_t = Activation('softmax', name='p_t')(p_t)
        r_s = WeightedAspectEmb(aspect_size, self._emb_dim, name='aspect_emb',
                                W_regularizer=self._ortho_reg)(p_t)

        # Loss
        loss = MaxMargin(neg_size=self._neg_size, name='max_margin')([z_s, z_n, r_s])
        model = Model(inputs=[emb_input, neg_input], outputs=loss)
        model.compile(optimizer=self._optimizer, loss=SimpleABAE.max_margin_loss, metrics=[SimpleABAE.max_margin_loss])

        return model        

class ABAE_T(SimpleABAE):
    '''
    This is a slight variation on ABAE, with a single-dimension secondary input (`target_input`).
    The target input is added to the positive weights layer following the Attention layer.
    '''
    def _compile_model(self):
        maxlen = self._n
        aspect_size = self._output_size

        # Inputs
        emb_input = Input(shape=(maxlen, self._emb_dim), dtype='float32', name='sentence_input')
        neg_input = Input(shape=(maxlen, self._emb_dim), dtype='float32', name='neg_input')
        target_input = Input(shape=(1), dtype='float32', name='target_input')

        # Compute sentence representation
        y_s = Average()(emb_input)
        att_weights = Attention_ABAE(name='att_weights')([emb_input, y_s])
        z_s = WeightedSum()([emb_input, att_weights])

        z_s = k_Average()([z_s, target_input]) # Integrates `target_input` into ABAE_T

        # Compute representations of negative instances
        z_n = Average()(neg_input)

        # Reconstruction
        p_t = Dense(aspect_size)(z_s)
        p_t = Activation('softmax', name='p_t')(p_t)
        r_s = WeightedAspectEmb(aspect_size, self._emb_dim, name='aspect_emb',
                                W_regularizer=self._ortho_reg)(p_t)

        # Loss
        loss = MaxMargin(neg_size=self._neg_size, name='max_margin')([z_s, z_n, r_s])
        model = Model(inputs=[emb_input, neg_input, target_input], outputs=loss)
        model.compile(optimizer=self._optimizer, loss=SimpleABAE.max_margin_loss, metrics=[SimpleABAE.max_margin_loss])

        return model
    
class ABAE_T_Emb(SimpleABAE_Emb):
    '''
    This is a slight variation on ABAE, with a single-dimension secondary input (`target_input`).
    The target input is added to the positive weights layer following the Attention layer.
    '''
    def _compile_model(self):
        maxlen = self._n
        aspect_size = self._output_size

        # Embedding layers
        embedding_layer = self._embedding_layer

        # Inputs
        emb_input = Input(shape=(maxlen,), dtype='float32', name='sentence_input')
        neg_input = Input(shape=(maxlen,), dtype='float32', name='neg_input')
        target_input = Input(shape=(1), dtype='float32', name='target_input')

        # Compute sentence representation
        pos_emb = embedding_layer(emb_input)
        y_s = Average()(pos_emb)
        att_weights = Attention_ABAE(name='att_weights')([pos_emb, y_s])
        z_s = WeightedSum()([pos_emb, att_weights])

        z_s = k_Average()([z_s, target_input]) # Integrates `target_input` into ABAE_T

        # Compute representations of negative instances
        neg_emb = embedding_layer(neg_input)
        z_n = Average()(neg_emb)

        # Reconstruction
        p_t = Dense(aspect_size)(z_s)
        p_t = Activation('softmax', name='p_t')(p_t)
        r_s = WeightedAspectEmb(aspect_size, self._emb_dim, name='aspect_emb',
                                W_regularizer=self._ortho_reg)(p_t)

        # Loss
        loss = MaxMargin(neg_size=self._neg_size, name='max_margin')([z_s, z_n, r_s])
        model = Model(inputs=[emb_input, neg_input, target_input], outputs=loss)
        model.compile(optimizer=self._optimizer, loss=SimpleABAE.max_margin_loss, metrics=[SimpleABAE.max_margin_loss])

        return model

class ABAE_O(SimpleABAE):
    '''
    This is a slight variation on ABAE, with an n-dimensional secondary input (`target_input`).
    The target input is averaged into the positive weights layer following the Attention layer.
    The secondary input is expected to be a 6-dimensional vector.
    '''
    def __init__(self, target_input_size, neg_size, win_size, emb_dim, output_size, optimizer='adam', loss='categorical_crossentropy', loss_weights=None):
        self._target_input_size = target_input_size

        super().__init__(neg_size, win_size, emb_dim, output_size, optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _compile_model(self):
        maxlen = self._n
        aspect_size = self._output_size

        # Inputs
        emb_input = Input(shape=(maxlen, self._emb_dim), dtype='float32', name='sentence_input')
        neg_input = Input(shape=(maxlen, self._emb_dim), dtype='float32', name='neg_input')
        target_input = Input(shape=(self._target_input_size), dtype='float32', name='target_input')

        # Compute sentence representation
        y_s = Average()(emb_input)
        att_weights = Attention_ABAE(name='att_weights')([emb_input, y_s])
        z_s = WeightedSum()([emb_input, att_weights])

        # Merge target_input into the model
        z_s = Concatenate()([z_s, target_input])
        z_s = Dense(self._emb_dim)(z_s)

        # Compute representations of negative instances
        z_n = Average()(neg_input)

        # Reconstruction
        p_t = Dense(aspect_size)(z_s)
        p_t = Activation('softmax', name='p_t')(p_t)
        r_s = WeightedAspectEmb(aspect_size, self._emb_dim, name='aspect_emb',
                                W_regularizer=self._ortho_reg)(p_t)

        # Loss
        loss = MaxMargin(neg_size=self._neg_size, name='max_margin')([z_s, z_n, r_s])
        model = Model(inputs=[emb_input, neg_input, target_input], outputs=loss)
        model.compile(optimizer=self._optimizer, loss=SimpleABAE.max_margin_loss, metrics=[SimpleABAE.max_margin_loss])

        return model

class ABAE_A(SimpleABAE):
    '''
    This is a slight variation on ABAE, with an n-dimensional secondary input (`target_input`).
    The target input is averaged into the positive weights layer following the Attention layer.
    '''
    def __init__(self, target_input_size, neg_size, win_size, emb_dim, output_size, optimizer='adam', loss='categorical_crossentropy', loss_weights=None):
        self._target_input_size = target_input_size

        super().__init__(neg_size, win_size, emb_dim, output_size, optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _compile_model(self):
        maxlen = self._n
        aspect_size = self._output_size

        # Inputs
        emb_input = Input(shape=(maxlen, self._emb_dim), dtype='float32', name='sentence_input')
        neg_input = Input(shape=(maxlen, self._emb_dim), dtype='float32', name='neg_input')
        target_input = Input(shape=(self._target_input_size), dtype='float32', name='target_input')

        # Compute sentence representation
        y_s = Average()(emb_input)
        att_weights = Attention_ABAE(name='att_weights')([emb_input, y_s])
        z_s = WeightedSum()([emb_input, att_weights])

        # Merge target_input into the model
        target_dense = Dense(self._emb_dim)(target_input)
        z_s = Attention()([z_s, target_dense])
        z_s = Dense(self._emb_dim)(z_s)

        # Compute representations of negative instances
        z_n = Average()(neg_input)

        # Reconstruction
        p_t = Dense(aspect_size)(z_s)
        p_t = Activation('softmax', name='p_t')(p_t)
        r_s = WeightedAspectEmb(aspect_size, self._emb_dim, name='aspect_emb',
                                W_regularizer=self._ortho_reg)(p_t)

        # Loss
        loss = MaxMargin(neg_size=self._neg_size, name='max_margin')([z_s, z_n, r_s])
        model = Model(inputs=[emb_input, neg_input, target_input], outputs=loss)
        model.compile(optimizer=self._optimizer, loss=SimpleABAE.max_margin_loss, metrics=[SimpleABAE.max_margin_loss])

        return model

class ABAE_ALSTM(SimpleABAE):
    '''
    This is a slight variation on ABAE, with an n-dimensional secondary input (`target_input`).
    The target input is averaged into the positive weights layer following the Attention layer.
    '''
    def __init__(self, target_input_size, neg_size, win_size, emb_dim, output_size, optimizer='adam', loss='categorical_crossentropy', loss_weights=None):
        self._target_input_size = target_input_size

        super().__init__(neg_size, win_size, emb_dim, output_size, optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _compile_model(self):
        maxlen = self._n
        aspect_size = self._output_size

        # Inputs
        emb_input = Input(shape=(maxlen, self._emb_dim), dtype='float32', name='sentence_input')
        neg_input = Input(shape=(maxlen, self._emb_dim), dtype='float32', name='neg_input')
        target_input = Input(shape=(self._target_input_size), dtype='float32', name='target_input')

        # Compute sentence representation
        y_s = Average()(emb_input)
        att_weights = Attention_ABAE(name='att_weights')([emb_input, y_s])
        z_s = WeightedSum()([emb_input, att_weights])

        # Merge target_input into the model
        target_rnn = LSTM(self._emb_dim)(target_input)
        z_s = Attention()([z_s, target_rnn])
        z_s = Dense(self._emb_dim)(z_s)

        # Compute representations of negative instances
        z_n = Average()(neg_input)

        # Reconstruction
        p_t = Dense(aspect_size)(z_s)
        p_t = Activation('softmax', name='p_t')(p_t)
        r_s = WeightedAspectEmb(aspect_size, self._emb_dim, name='aspect_emb',
                                W_regularizer=self._ortho_reg)(p_t)

        # Loss
        loss = MaxMargin(neg_size=self._neg_size, name='max_margin')([z_s, z_n, r_s])
        model = Model(inputs=[emb_input, neg_input, target_input], outputs=loss)
        model.compile(optimizer=self._optimizer, loss=SimpleABAE.max_margin_loss, metrics=[SimpleABAE.max_margin_loss])

        return model