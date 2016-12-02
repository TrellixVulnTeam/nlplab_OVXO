import functools
import numpy as np
import math
import tensorflow as tf
import sandbox.rocky.tf.core.layers as L

class GRULayer(L.Layer):
    """
    A gated recurrent unit implements the following update mechanism:
    Reset gate:        r(t) = f_r(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
    Update gate:       u(t) = f_u(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
    Cell gate:         c(t) = f_c(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
    New hidden state:  h(t) = (1 - u(t)) * h(t-1) + u_t * c(t)
    Note that the reset, update, and cell vectors must have the same dimension as the hidden state
    """

    def __init__(self, incoming, num_units, hidden_nonlinearity,
                 gate_nonlinearity=tf.nn.sigmoid, W_x_init=L.XavierUniformInitializer(), W_h_init=L.OrthogonalInitializer(),
                 b_init=tf.zeros_initializer, hidden_init=tf.zeros_initializer, hidden_init_trainable=False,
                 layer_normalization=False, add_h0=True, **kwargs):

        if hidden_nonlinearity is None:
            hidden_nonlinearity = tf.identity

        if gate_nonlinearity is None:
            gate_nonlinearity = tf.identity

        super(GRULayer, self).__init__(incoming, **kwargs)

        input_shape = self.input_shape[2:]

        input_dim = np.prod(input_shape)

        self.layer_normalization = layer_normalization

        # Weights for the initial hidden state
        if add_h0:
            self.h0 = self.add_param(hidden_init, (num_units,), name="h0", trainable=hidden_init_trainable,
                                     regularizable=False)

        # Weights for the reset gate
        self.W_xr = self.add_param(W_x_init, (input_dim, num_units), name="W_xr")
        self.W_hr = self.add_param(W_h_init, (num_units, num_units), name="W_hr")
        self.b_r = self.add_param(b_init, (num_units,), name="b_r", regularizable=False)
        # Weights for the update gate
        self.W_xu = self.add_param(W_x_init, (input_dim, num_units), name="W_xu")
        self.W_hu = self.add_param(W_h_init, (num_units, num_units), name="W_hu")
        self.b_u = self.add_param(b_init, (num_units,), name="b_u", regularizable=False)
        # Weights for the cell gate
        self.W_xc = self.add_param(W_x_init, (input_dim, num_units), name="W_xc")
        self.W_hc = self.add_param(W_h_init, (num_units, num_units), name="W_hc")
        self.b_c = self.add_param(b_init, (num_units,), name="b_c", regularizable=False)

        self.W_x_ruc = tf.concat(1, [self.W_xr, self.W_xu, self.W_xc])
        self.W_h_ruc = tf.concat(1, [self.W_hr, self.W_hu, self.W_hc])
        self.W_x_ru = tf.concat(1, [self.W_xr, self.W_xu])
        self.W_h_ru = tf.concat(1, [self.W_hr, self.W_hu])
        self.b_ruc = tf.concat(0, [self.b_r, self.b_u, self.b_c])

        self.gate_nonlinearity = gate_nonlinearity
        self.num_units = num_units
        self.nonlinearity = hidden_nonlinearity
        self.norm_params = dict()

        # pre-run the step method to initialize the normalization parameters

        # we are not using layer normalization ...
        #h_dummy = tf.placeholder(dtype=tf.float32, shape=(None, num_units), name="h_dummy")
        #x_dummy = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name="x_dummy")
        #self.step(h_dummy, x_dummy)

    def step(self, hprev, x):
        if self.layer_normalization:
            ln = L.apply_ln(self)
            x_ru = ln(tf.matmul(x, self.W_x_ru), "x_ru")
            h_ru = ln(tf.matmul(hprev, self.W_h_ru), "h_ru")
            x_r, x_u = tf.split(split_dim=1, num_split=2, value=x_ru)
            h_r, h_u = tf.split(split_dim=1, num_split=2, value=h_ru)
            x_c = ln(tf.matmul(x, self.W_xc), "x_c")
            h_c = ln(tf.matmul(hprev, self.W_hc), "h_c")
            r = self.gate_nonlinearity(x_r + h_r)
            u = self.gate_nonlinearity(x_u + h_u)
            c = self.nonlinearity(x_c + r * h_c)
            h = (1 - u) * hprev + u * c
            return h
        else:
            xb_ruc = tf.matmul(x, self.W_x_ruc) + tf.reshape(self.b_ruc, (1, -1))
            h_ruc = tf.matmul(hprev, self.W_h_ruc)
            xb_r, xb_u, xb_c = tf.split(split_dim=1, num_split=3, value=xb_ruc)
            h_r, h_u, h_c = tf.split(split_dim=1, num_split=3, value=h_ruc)
            r = self.gate_nonlinearity(xb_r + h_r)
            u = self.gate_nonlinearity(xb_u + h_u)
            c = self.nonlinearity(xb_c + r * h_c)
            h = (1 - u) * hprev + u * c
            return h

    def get_step_layer(self, l_in, l_prev_hidden, name=None):
        return L.GRUStepLayer(incomings=[l_in, l_prev_hidden], recurrent_layer=self, name=name)

    def get_output_shape_for(self, input_shape):
        n_batch, n_steps = input_shape[:2]
        return n_batch, n_steps, self.num_units

    def get_output_for(self, input, **kwargs):
        input_shape = tf.shape(input)
        n_batches = input_shape[0]
        n_steps = input_shape[1]
        input = tf.reshape(input, tf.pack([n_batches, n_steps, -1]))
        # print "h0 shape: ", self.h0.get_shape()

        if 'recurrent_state' in kwargs and self in kwargs['recurrent_state']:
            h0s = kwargs['recurrent_state'][self]
        else:
            h0s = tf.tile(
                tf.reshape(self.h0, (1, self.num_units)),
                (n_batches, 1)
            )
            # print "h0s shape: ", h0s
        # flatten extra dimensions
        shuffled_input = tf.transpose(input, (1, 0, 2))
        hs = tf.scan(
            self.step,
            elems=shuffled_input,
            initializer=h0s
        )
        shuffled_hs = tf.transpose(hs, (1, 0, 2))
        if 'recurrent_state_output' in kwargs:
            kwargs['recurrent_state_output'][self] = shuffled_hs
        return shuffled_hs

class AttnGRULayer(GRULayer):
    def __init__(self, incoming,
                 W_hs_init=L.XavierUniformInitializer(),
                 W_gamma_init=L.XavierUniformInitializer(),
                 W_concat_init=L.XavierUniformInitializer(),
                 b_init = tf.zeros_initializer,
                 **kwargs):

        max_length, n_env, num_units = kwargs["encoder_max_seq_length"], kwargs["n_env"], kwargs["num_units"]

        # instantiate GRULayer
        super(AttnGRULayer, self).__init__(incoming, add_h0=False, **kwargs)

        with tf.variable_scope("attn"):
            self.W_hs = self.add_param(W_hs_init, (num_units, num_units), name="W_hs")
            self.b_hs = self.add_param(b_init, (num_units,), name="b_c", regularizable=False)

            self.W_gamma = self.add_param(W_gamma_init, (num_units, num_units), name="W_gamma")
            self.b_gamma = self.add_param(b_init, (num_units,), name="b_gamma")

            self.W_concat = self.add_param(W_concat_init, (2 * num_units, num_units), name="W_concat")
            self.b_concat = self.add_param(b_init, (num_units,), name="b_concat")

        self.hs = tf.placeholder(tf.float32, shape=(max_length, n_env, num_units), name="hs")  # pass this in
        self.h0 = self.hs[-1,:,:] # last timestep, every environment.
        self.h0_sym = tf.placeholder(tf.float32, shape=(n_env, num_units), name="h0_sym")
        # for the get_output_for() method

        # print "attn hs shape: ", self.hs.get_shape()
        # print "attn h0 shape: ", self.h0.get_shape()

        hs2d = tf.reshape(self.hs, [-1, num_units])
        phi_hs2d = tf.nn.tanh(tf.nn.xw_plus_b(hs2d, self.W_hs, self.b_hs))
        self.phi_hs = tf.reshape(phi_hs2d, tf.shape(self.hs))


    def step(self, hprev, x):
        gru_out = super(AttnGRULayer, self).step(hprev,x)
        # do attention based post-processing using hs.

        gamma_h = tf.nn.tanh(tf.nn.xw_plus_b(gru_out, self.W_gamma, self.b_gamma))
        weights = tf.reduce_sum(self.phi_hs * gamma_h, reduction_indices=2, keep_dims=True)
        weights = tf.exp(weights - tf.reduce_max(weights, reduction_indices=0, keep_dims=True))
        weights = weights / (1e-6 + tf.reduce_sum(weights, reduction_indices=0, keep_dims=True))
        context = tf.reduce_sum(self.hs * weights, reduction_indices=0)

        out = tf.nn.relu(tf.nn.xw_plus_b(tf.concat(1,[context, gru_out]), self.W_concat, self.b_concat))

        _slice_map = tf.slice(weights, [0, 0, 0], [-1, -1, 1])
        self.attn_map = tf.squeeze(_slice_map)

        return out

    # this doesn't work. It's called by L.getoutput()
    def get_output_for(self, input, **kwargs):
        input_shape = tf.shape(input)
        n_batches = input_shape[0]
        n_steps = input_shape[1]
        input = tf.reshape(input, tf.pack([n_batches, n_steps, -1]))
        # print "h0 shape: ", self.h0.get_shape()  # (12, 4)

        if 'recurrent_state' in kwargs and self in kwargs['recurrent_state']:
            h0s = kwargs['recurrent_state'][self]
            # we must use this because tf.scan can't work with placeholder slice
            # for some reason (at least not on my macbook)
            # L.get_output([],
            #  recurrent_state={prob_network.recurrent_layer: np.ones([12, 4], dtype="float32")}), assign outside
        else:
            # reshape: from (12, 4) to (1, 4), no need

            # For attentional layer, we should just make h0s = self.h0
            # the below commented code takes ONE h0 and expand it to all instances in the batch.
            # we don't need because our h0 is different for every batch

            # h0s = tf.tile(
            #     tf.reshape(self.h0, (1, self.num_units)),
            #     (n_batches, 1)
            # )
            # h0s = tf.zeros(self.h0.get_shape(), name='h0s')
            # this is broken :(
            # h0s_temp = tf.get_variable("h0s", shape=self.h0.get_shape(), initializer=tf.zeros_initializer)
            # h0s = h0s_temp.assign(self.h0)

            # weirdly current this works for VPG
            h0s = self.h0_sym

        # flatten extra dimensions
        shuffled_input = tf.transpose(input, (1, 0, 2))
        hs = tf.scan(
            self.step,
            elems=shuffled_input,
            initializer=h0s
        )
        # there is literally no point to use tf.scan....because it's just 1 timestep

        shuffled_hs = tf.transpose(hs, (1, 0, 2))
        if 'recurrent_state_output' in kwargs:
            kwargs['recurrent_state_output'][self] = shuffled_hs
        return shuffled_hs
