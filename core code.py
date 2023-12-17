import tensorflow as tf
import numpy as np
from keras.layers import Layer
from keras import backend as K
from keras.layers import Dense

class BILinearAttention(Layer):
    def __init__(self, factor, dropout):
        super(BILinearAttention, self).__init__()
        self.hidden_factor = factor
        self.k = 8
        self.w1_reg = 1e-4
        self.v_reg = 1e-4
        self.attention = True
        self.dropout = dropout

    def build(self, input_shape):
        self.w1 = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                  initializer=tf.random_normal_initializer(),
                                  trainable=True,
                                  regularizer=tf.keras.regularizers.l2(self.w1_reg))
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True, )
        self.attention_w = self.add_weight(name='attention_w', shape=(self.hidden_factor, self.k),
                                           initializer=tf.random_normal_initializer(),
                                           trainable=True,
                                           regularizer=tf.keras.regularizers.l2(self.v_reg))
        self.attention_b = self.add_weight(name='attention_b', shape=(1, self.k),
                                           initializer=tf.zeros_initializer(),
                                           trainable=True)
        self.attention_p = self.add_weight(name='attention_p', shape=(1, self.k),
                                           initializer=tf.random_normal_initializer(),
                                           trainable=True,
                                           regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs, **kwargs):
        emb = inputs
        output_linear = tf.matmul(emb, self.w1) + self.w0 
        output_linear = tf.reduce_sum(output_linear, axis=1, keep_dims=False)
        summed_features_emb_square = tf.square(summed_features_emb) 
        squared_features_emb = tf.square(emb)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # (None ,self.hidden_units_mlp[0])
        outputs_bi = 0.5 * tf.subtract(summed_features_emb_square,
                                       squared_sum_features_emb)  # (None,self.hidden_units_mlp[0]=128)
        if self.attention:
            attention_mul = tf.reshape(
                tf.matmul(tf.reshape(outputs_bi, shape=[-1, self.hidden_factor]), self.attention_w),
                shape=[-1, self.k]) 
            attention_relu = tf.reduce_sum(
                tf.multiply(self.attention_p, tf.nn.relu(attention_mul + self.attention_b)), 1,
                keep_dims=True) 
            attention_out = tf.nn.softmax(attention_relu)
            attention_out = tf.nn.dropout(attention_out, self.dropout) 
        if self.attention:
            output_biattention = tf.multiply(attention_out, outputs_bi)
        else:
            output_biattention = outputs_bi 

        output = output_biattention + output_linear
        return output


class BILinear(Layer):
    def __init__(self):
        super(BILinear, self).__init__()
        self.w1_reg = 1e-4

    def build(self, input_shape):
        self.w1 = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                  initializer=tf.random_normal_initializer(),
                                  trainable=True,
                                  regularizer=tf.keras.regularizers.l2(self.w1_reg))
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True, )

    def call(self, inputs, **kwargs):
        emb = inputs
        output_linear = tf.matmul(emb, self.w1) + self.w0 
        output_linear = tf.reduce_sum(output_linear, axis=1, keep_dims=False)
        summed_features_emb = tf.reduce_sum(emb, 1)  # (None, self.hidden_units_mlp[0])
        summed_features_emb_square = tf.square(summed_features_emb)
        squared_features_emb = tf.square(emb)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # (None ,self.hidden_units_mlp[0])

        outputs_bi = 0.5 * tf.subtract(summed_features_emb_square,
                                       squared_sum_features_emb)  # (None,self.hidden_units_mlp[0]=128)

        output_biattention = outputs_bi  # (None,self.hidden_factor=128)
        output = output_biattention + output_linear
        return output

class Pearson_corr(Layer):

    def __init__(self, input2, **kwargs):
        super(Pearson_corr, self).__init__(**kwargs)
        self.input2 = input2

    def call(self, input1, **kwargs):
        feature1 = input1
        feature2 = self.input2
        mean1, variance1 = tf.nn.moments(feature1, axes=[0])
        mean2, variance2 = tf.nn.moments(feature2, axes=[0])
        epsilon = 1e-8
        stddev1 = tf.sqrt(variance1 + epsilon)
        stddev2 = tf.sqrt(variance2 + epsilon)
        norm1 = (feature1 - mean1) / stddev1
        norm2 = (feature2 - mean2) / stddev2
        mean_feature1 = tf.reduce_mean(norm1)
        mean_feature2 = tf.reduce_mean(norm2)
        numerator = np.sum((norm1 - mean_feature1) * (norm2 - mean_feature2))
        denominator = tf.math.sqrt(np.sum((norm1 - mean_feature1) ** 2) * np.sum((norm2 - mean_feature2) ** 2))
        correlation = numerator / denominator
        return correlation * feature2


class Cosine_similiarity(Layer):

    def __init__(self, input2, **kwargs):
        super(Cosine_similiarity, self).__init__(**kwargs)
        self.input2 = input2

    def call(self, input1, **kwargs):
        feature1 = input1
        feature2 = self.input2
        mean1, variance1 = tf.nn.moments(feature1, axes=[0])
        mean2, variance2 = tf.nn.moments(feature2, axes=[0])
        epsilon = 1e-8
        stddev1 = tf.sqrt(variance1 + epsilon)
        stddev2 = tf.sqrt(variance2 + epsilon)
        # norm
        norm1 = (feature1 - mean1) / stddev1
        norm2 = (feature2 - mean2) / stddev2
        dot_product = np.dot(norm1, norm2)
        norm_feature1 = tf.norm(norm1)
        norm_feature2 = tf.norm(norm2)
        similarity = dot_product / (norm_feature1 * norm_feature2)
        return similarity * feature2

class DotProductAttention(Layer):
    def __init__(self, dropout=0.0):
        super(DotProductAttention, self).__init__()
        self.dropout = dropout

    def call(self, inputs, **kwargs):
        queries, keys, values = inputs  
        score = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  
        score = score / int(queries.shape[-1]) ** 0.5
        score = K.softmax(score)  
        outputs = K.batch_dot(score, values)  
        return outputs
