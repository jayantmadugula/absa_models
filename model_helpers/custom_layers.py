''' 
This file contains custom Keras/Tensorflow 2 layers.

References:
The Attention_ABAE, Average, MaxMargin, WeightedAspectEmb, and WeightedSum layers were based on code from the following repositories:
- https://github.com/ruidan/Unsupervised-Aspect-Extraction/blob/master/code/my_layers.py
- https://github.com/luckmoon/Unsupervised-Aspect-Extraction/blob/master/code/custom_layers.py
'''

from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers, constraints, initializers
import tensorflow.keras.backend as K


class Attention_ABAE(Layer):
    ''' 
    This class defines the custom Attention Layer from He et al.'s "An Unsupervised Neural Attention Model for Aspect Extraction".
    '''
    def __init__(self, W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias

        super(Attention_ABAE, self).__init__(**kwargs)
        
    def build(self, input_shape):
        assert(type(input_shape) == list)
        assert(len(input_shape) == 2) # Ensure only two arguments are being passed in
        
        self.steps = input_shape[0][1]
        self.W = self.add_weight(
            shape=[int(input_shape[0][-1]), int(input_shape[1][-1])], 
            initializer='glorot_uniform', 
            name='{}_W'.format(self.name), 
            regularizer=self.W_regularizer, 
            constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(
                shape=(1,), 
                initializer='zero', 
                name='{}_b'.format(self.name), 
                regularizer=self.b_regularizer, 
                constraint=self.b_constraint)

        self.built = True
        
    def call(self, input_tensor, mask=None):
        x = input_tensor[0]
        y = input_tensor[1]
        # mask = mask[0]
        
        y = K.transpose(K.dot(self.W, K.transpose(y)))
        y = K.expand_dims(y, axis=-2)
        y = K.repeat_elements(y, self.steps, axis=1)
        eij = K.sum(x * y, axis=-1)

        if self.bias:
            b = K.repeat_elements(self.b, self.steps, axis=0)
            eij += b

        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        return a

    def compute_mask(self, input_tensor, mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return input_shape[0][0], input_shape[0][1]

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)


class WeightedSum(Layer):
    ''' 
    A custom layer that computes a weighted sum of an input vector.
    This layer takes two inputs, the vector to be summed and the weights.
    These two input vectors must have the same length.
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightedSum, self).__init__(**kwargs) 
        
    def call(self, input_tensor, mask=None):
        assert(type(input_tensor) == list)

        x = input_tensor[0]
        a = input_tensor[1]
        
        a = K.expand_dims(a)
        weighted_tensor = x * a
        
        return K.sum(weighted_tensor, axis=1)

    def compute_mask(self, input_tensor, mask=None):
        return None
        
    def get_output_shape_for(self, input_shape):
        return input_shape[0][0], input_shape[0][-1]

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)
        
        
class WeightedAspectEmb(Layer):
    '''
    A custom layer that implements a weighted Aspect Embedding reconstruction from He et al.'s "An Unsupervised Neural Attention Model for Aspect Extraction".
    '''
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        init='uniform', 
        input_length=None, 
        W_regularizer=None, 
        activity_regularizer=None, 
        W_constraint=None, 
        weights=None, 
        dropout=0., 
        **kwargs):
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializers.get(init)
        self.input_length = input_length
        self.dropout = dropout

        self.W_constraint = constraints.get(W_constraint)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        self.initial_weights = weights
        kwargs['input_shape'] = (self.input_length,)
        kwargs['dtype'] = K.floatx()

        super(WeightedAspectEmb, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.input_dim, self.output_dim), initializer=self.init, name='{}_W'.format(self.name), regularizer=self.W_regularizer, constraint=self.W_constraint)
        
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)

        self.built = True

    def compute_mask(self, x, mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)


class Average(Layer):
    ''' A basic layer that computes the average of an inputted vector. '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Average, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        return K.mean(x, axis=-2)
    
    def get_output_shape_for(self, input_shape):
        return input_shape[0:-2]+input_shape[-1:]
    
    def compute_mask(self, x, mask=None):
        return None


class MaxMargin(Layer):
    ''' 
    This custom layer is used to compute loss in He et al.'s "An Unsupervised Neural Attention Model for Aspect Extraction".
    '''
    def __init__(self, neg_size, **kwargs):
        super(MaxMargin, self).__init__(**kwargs)
        self.neg_size = neg_size

    def call(self, input_tensor, mask=None):
        z_s = input_tensor[0]
        z_n = input_tensor[1]
        r_s = input_tensor[2]
        
        z_s = K.l2_normalize(z_s, axis=-1)
        z_n = K.l2_normalize(z_n, axis=-1)
        r_s = K.l2_normalize(r_s, axis=-1)

        steps = K.int_shape(z_n)
        pos = K.sum(r_s * z_s, axis=-1, keepdims=True) # pos.shape == (batch_size, 1)
        neg = K.dot(r_s, K.transpose(z_n)) # neg.shape == (batch_size, neg_size)
        loss = K.cast(K.sum(K.maximum(0., (1. - pos + neg)), axis=-1, keepdims=True), K.floatx())

        return loss

    def compute_mask(self, input_tensor, mask=None):
        return None
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)