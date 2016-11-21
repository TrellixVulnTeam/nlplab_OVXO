"""
Neural net encoder/decoder (seperate)
(Not sure if I still want this...)
"""

from __future__ import absolute_import
from __future__ import division

import random

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

import exps.nlc_env


class GRUCellAttn(rnn_cell.GRUCell):
    def __init__(self, num_units, encoder_output, scope=None):
        self.hs = encoder_output
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn1"):
                hs2d = tf.reshape(self.hs, [-1, num_units])
                phi_hs2d = tanh(rnn_cell._linear(hs2d, num_units, True, 1.0))
                self.phi_hs = tf.reshape(phi_hs2d, tf.shape(self.hs))
        super(GRUCellAttn, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        gru_out, gru_state = super(GRUCellAttn, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn2"):
                gamma_h = tanh(rnn_cell._linear(gru_out, self._num_units, True, 1.0))
            weights = tf.reduce_sum(self.phi_hs * gamma_h, reduction_indices=2, keep_dims=True)
            weights = tf.exp(weights - tf.reduce_max(weights, reduction_indices=0, keep_dims=True))
            weights = weights / (1e-6 + tf.reduce_sum(weights, reduction_indices=0, keep_dims=True))
            context = tf.reduce_sum(self.hs * weights, reduction_indices=0)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))
            self.attn_map = tf.squeeze(tf.slice(weights, [0, 0, 0], [-1, -1, 1]))
            return (out, out)


class Encoder(object):
    def __init__(self, vocab_size, size, num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, dropout, forward_only=False):

        self.size = size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.keep_prob_config = 1.0 - dropout
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        self.keep_prob = tf.placeholder(tf.float32)
        self.source_tokens = tf.placeholder(tf.int32, shape=[None, None])
        self.target_tokens = tf.placeholder(tf.int32, shape=[None, None])
        self.source_mask = tf.placeholder(tf.int32, shape=[None, None])
        self.target_mask = tf.placeholder(tf.int32, shape=[None, None])
        self.beam_size = tf.placeholder(tf.int32)
        self.target_length = tf.reduce_sum(self.target_mask, reduction_indices=0)

        self.decoder_state_input, self.decoder_state_output = [], []
        for i in xrange(num_layers):
            self.decoder_state_input.append(tf.placeholder(tf.float32, shape=[None, size]))

        with tf.variable_scope("Encoder", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_encoder()

            # calculate/setup loss later

            # self.setup_decoder()
            # self.setup_loss()
            # self.setup_beam()

            # params = tf.trainable_variables()
            # if not forward_only:
            #     opt = tf.train.AdamOptimizer(self.learning_rate)
            #
            #     gradients = tf.gradients(self.losses, params)
            #     clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
            #     # self.gradient_norm = tf.global_norm(clipped_gradients)
            #     self.gradient_norm = tf.global_norm(gradients)
            #     self.param_norm = tf.global_norm(params)
            #     self.updates = opt.apply_gradients(
            #         zip(clipped_gradients, params), global_step=self.global_step)
            #
            # self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

    def setup_embeddings(self):
        with vs.variable_scope("embeddings"):
            self.L_enc = tf.get_variable("L_enc", [self.vocab_size, self.size])
            # self.L_dec = tf.get_variable("L_dec", [self.vocab_size, self.size])
            self.encoder_inputs = embedding_ops.embedding_lookup(self.L_enc, self.source_tokens)
            # self.decoder_inputs = embedding_ops.embedding_lookup(self.L_dec, self.target_tokens)

    def setup_encoder(self):
        self.encoder_cell = rnn_cell.GRUCell(self.size)
        with vs.variable_scope("PryamidEncoder"):
            inp = self.encoder_inputs
            mask = self.source_mask
            out = None
            for i in xrange(self.num_layers):
                with vs.variable_scope("EncoderCell%d" % i) as scope:
                    srclen = tf.reduce_sum(mask, reduction_indices=0)
                    out, _ = self.bidirectional_rnn(self.encoder_cell, inp, srclen, scope=scope)
                    dropin, mask = self.downscale(out, mask)
                    inp = self.dropout(dropin)
            self.encoder_output = out

    def dropout(self, inp):
        return tf.nn.dropout(inp, self.keep_prob)

    def downscale(self, inp, mask):
        with vs.variable_scope("Downscale"):
            inshape = tf.shape(inp)
            T, batch_size, dim = inshape[0], inshape[1], inshape[2]
            inp2d = tf.reshape(tf.transpose(inp, perm=[1, 0, 2]), [-1, 2 * self.size])
            out2d = rnn_cell._linear(inp2d, self.size, True, 1.0)
            out3d = tf.reshape(out2d, tf.pack((batch_size, tf.to_int32(T / 2), dim)))
            out3d = tf.transpose(out3d, perm=[1, 0, 2])
            out3d.set_shape([None, None, self.size])
            out = tanh(out3d)

            mask = tf.transpose(mask)
            mask = tf.reshape(mask, [-1, 2])
            mask = tf.cast(mask, tf.bool)
            mask = tf.reduce_any(mask, reduction_indices=1)
            mask = tf.to_int32(mask)
            mask = tf.reshape(mask, tf.pack([batch_size, -1]))
            mask = tf.transpose(mask)

        return out, mask

    def bidirectional_rnn(self, cell, inputs, lengths, scope=None):
        name = scope.name or "BiRNN"
        # Forward direction
        with vs.variable_scope(name + "_FW") as fw_scope:
            output_fw, output_state_fw = rnn.dynamic_rnn(cell, inputs, time_major=True, dtype=dtypes.float32,
                                                         sequence_length=lengths, scope=fw_scope)
        # Backward direction
        inputs_bw = tf.reverse_sequence(inputs, tf.to_int64(lengths), seq_dim=0, batch_dim=1)
        with vs.variable_scope(name + "_BW") as bw_scope:
            output_bw, output_state_bw = rnn.dynamic_rnn(cell, inputs_bw, time_major=True, dtype=dtypes.float32,
                                                         sequence_length=lengths, scope=bw_scope)

        output_bw = tf.reverse_sequence(output_bw, tf.to_int64(lengths), seq_dim=0, batch_dim=1)

        outputs = output_fw + output_bw
        output_state = output_state_fw + output_state_bw

        return (outputs, output_state)


class Decoder(object):
    def __init__(self, vocab_size, size, num_layers, encoder_output, batch_size, dropout):
        self.size = size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.keep_prob_config = 1.0 - dropout
        self.encoder_output = encoder_output

        self.target_tokens = tf.placeholder(tf.int32, shape=[None, None])
        self.target_mask = tf.placeholder(tf.int32, shape=[None, None])
        self.keep_prob = tf.placeholder(tf.float32)

        self.target_length = tf.reduce_sum(self.target_mask, reduction_indices=0)

        self.decoder_state_input, self.decoder_state_output = [], []
        for i in xrange(num_layers):
            self.decoder_state_input.append(tf.placeholder(tf.float32, shape=[None, size]))

    def dropout(self, inp):
        return tf.nn.dropout(inp, self.keep_prob)

    def setup_embeddings(self):
        with vs.variable_scope("embeddings"):
            self.L_dec = tf.get_variable("L_dec", [self.vocab_size, self.size])
            self.decoder_inputs = embedding_ops.embedding_lookup(self.L_dec, self.target_tokens)

    def setup_decoder(self):
        if self.num_layers > 1:
            self.decoder_cell = rnn_cell.GRUCell(self.size)
        self.attn_cell = GRUCellAttn(self.size, self.encoder_output, scope="DecoderAttnCell")

        with vs.variable_scope("Decoder"):
            inp = self.decoder_inputs
            for i in xrange(self.num_layers - 1):
                with vs.variable_scope("DecoderCell%d" % i) as scope:
                    out, state_output = rnn.dynamic_rnn(self.decoder_cell, inp, time_major=True,
                                                        dtype=dtypes.float32, sequence_length=self.target_length,
                                                        scope=scope, initial_state=self.decoder_state_input[i])
                    inp = self.dropout(out)
                    self.decoder_state_output.append(state_output)

            with vs.variable_scope("DecoderAttnCell") as scope:
                out, state_output = rnn.dynamic_rnn(self.attn_cell, inp, time_major=True,
                                                    dtype=dtypes.float32, sequence_length=self.target_length,
                                                    scope=scope, initial_state=self.decoder_state_input[i + 1])
                self.decoder_output = self.dropout(out)
                self.decoder_state_output.append(state_output)
