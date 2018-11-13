from keras.layers import Embedding, LSTM, Bidirectional, Dense, TimeDistributed, Dropout, Layer, Input, Activation
from keras import backend as K

class AttentionLayer(Layer):
    '''ref: https://blog.csdn.net/uhauha2929/article/details/80733255 '''

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        print('input_shape:', input_shape)
        assert len(input_shape)==3
        #W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        print('inputs.shape:', inputs.shape)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        print('x.shape:', x.shape)
        print('W.shape:', self.W.shape)
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        print('outputs.shape', outputs.shape)
        return outputs

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
