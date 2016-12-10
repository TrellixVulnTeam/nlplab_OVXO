def test_gru_network():
    from rllab.core.network import GRUNetwork
    import lasagne.layers as L
    from rllab.misc import ext
    import numpy as np
    network = GRUNetwork(
        input_shape=(2, 3),
        output_dim=5,
        hidden_dim=4,
    )
    f_output = ext.compile_function(
        inputs=[network.input_layer.input_var],
        outputs=L.get_output(network.output_layer)
    )
    assert f_output(np.zeros((6, 8, 2, 3))).shape == (6, 8, 5)


def test_tf_gru_network():
    from sandbox.rocky.tf.core.network import GRUNetwork
    import numpy as np
    import tensorflow as tf
    from sandbox.rocky.tf.misc import tensor_utils
    import sandbox.rocky.tf.core.layers as L
    from exps.layers import AttnGRULayer, GRULayer

    sess = tf.InteractiveSession()
    input_dim = 3
    # prob_network = GRUNetwork(
    #     name="prob_network",
    #     input_shape=(3,),  # feature_dim
    #     output_dim=5,
    #     hidden_dim=4,
    #     gru_layer_cls=GRULayer
    # )
    #
    # f_output = tensor_utils.compile_function(
    #     inputs=[
    #         # flat_input_var,
    #         prob_network.input_layer.input_var
    #         # network.step_prev_state_layer.input_var,
    #     ],
    #     outputs=L.get_output(
    #         prob_network.output_layer
    #     )
    # )
    # sess.run(tf.initialize_all_variables())
    # # for network.input_layer.input_var, we must pass in: (?, ?, 3)
    #
    # # we can use np.shae() to expand
    # assert f_output(np.zeros((12, 1, 3), dtype="float32")).shape == (12, 1, 5)
    # # it works

    # ======= test: attention cell (should work fine as well) ========

    # prob_network = GRUNetwork(
    #     name="prob_network",
    #     input_shape=(3,),  # feature_dim
    #     output_dim=5,
    #     hidden_dim=4,
    #     gru_layer_cls=AttnGRULayer,
    #     layer_args={"max_length": 100,
    #                 "n_env": 12}
    # )
    #
    # # must do this or Python won't work
    # # rec_state = dict()
    # # rec_state['recurrent_state'] = {}
    # # rec_state['recurrent_state'][prob_network.recurrent_layer] = np.ones([12, 4])
    #
    # f_output = tensor_utils.compile_function(
    #     inputs=[
    #         prob_network.input_layer.input_var,
    #         prob_network.recurrent_layer.hs
    #     ],
    #     outputs=L.get_output(
    #         prob_network.output_layer,
    #         # explicitly pass in last hidden layers
    #         recurrent_state={prob_network.recurrent_layer: np.ones([12, 4], dtype="float32")}
    #     )
    # )
    # sess.run(tf.initialize_all_variables())
    # # for network.input_layer.input_var, we must pass in: (?, ?, 3)
    #
    # # we can use np.shae() to expand
    # # num_units is 5..hidden units (should be)
    # print f_output(np.zeros((12, 1, 3), dtype="float32"), np.zeros([100, 12, 4])).shape
    #
    # assert f_output(np.zeros((12, 1, 3), dtype="float32"), np.zeros([100, 12, 4])).shape == (12, 1, 5)
    # # test passed!


    # ============= test: f_step_prob =============

    # prob_network = GRUNetwork(
    #     name="prob_network",
    #     input_shape=(3,),  # feature_dim/word_emb dim
    #     output_dim=5,  # this should match action_space
    #     hidden_dim=4,
    #     gru_layer_cls=AttnGRULayer,
    #     layer_args={"max_length": 100,
    #                 "n_env": 12}
    # )
    #
    # # so this is (n_batch, input_dim)?
    # flat_input_var = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name="flat_input")
    # feature_var = flat_input_var
    #
    # # get this from encoder's hs
    # hs = np.ones([100, 12, 4])
    # batched_h0 = hs[-1,:,:]
    #
    # f_step_prob = tensor_utils.compile_function(
    #     inputs=[
    #         flat_input_var,
    #         prob_network.step_prev_state_layer.input_var,
    #         prob_network.recurrent_layer.hs
    #     ],
    #     outputs=L.get_output([
    #         prob_network.step_output_layer,
    #         prob_network.step_hidden_layer
    #     ], inputs={prob_network.step_input_layer: feature_var},
    #     )
    # )
    #
    # sess.run(tf.initialize_all_variables())
    #
    # # it works!
    # print f_step_prob(np.zeros((12, 3)), batched_h0, hs)[0].shape == (12, 5)

    # ============== test: derivative of test 2, test 1 batch logit generation =====
    # we assign weights in them

    # 1st: assign weights 0

    prob_network = GRUNetwork(
        name="prob_network",
        input_shape=(3,),  # feature_dim
        output_dim=1,
        hidden_dim=4,
        gru_layer_cls=AttnGRULayer,
        layer_args={"encoder_max_seq_length": 30,
                    "n_env": 1}
    )

    # must do this or Python won't work
    # rec_state = dict()
    # rec_state['recurrent_state'] = {}
    # rec_state['recurrent_state'][prob_network.recurrent_layer] = np.ones([12, 4])

    prob_network.recurrent_layer.assign_hs_weights(np.zeros([30, 1, 4]))

    f_output = tensor_utils.compile_function(
        inputs=[
            prob_network.input_layer.input_var,
            # prob_network.recurrent_layer.hs,
            # prob_network.recurrent_layer.h0_sym
        ],
        outputs=L.get_output(
            prob_network.output_layer,
            # explicitly pass in last hidden layers
            # recurrent_state={prob_network.recurrent_layer: np.ones([1, 4], dtype="float32")}
        )
    )
    sess.run(tf.initialize_all_variables())
    # for network.input_layer.input_var, we must pass in: (?, ?, 3)

    # hs should be (encoder_max_length, n_env, num_units)
    # n_env is also the "batch_size", which is VERY important for the policy (actor)
    # but for the critic (Q-approx), since we always pass in paths[i], it's always batch_size 1
    print f_output(np.zeros((1, 32, 3), dtype="float32")).shape
    print f_output(np.zeros((1, 32, 3), dtype="float32"))

    # assert f_output(np.zeros((1, 32, 3), dtype="float32"), np.zeros([30, 1, 4]), np.ones([1, 4], dtype="float32")).shape == (1, 32, 1)
    # test passed!

    # 2nd: assign weights 1

    prob_network.recurrent_layer.assign_hs_weights(np.ones([30, 1, 4]))
    print f_output(np.zeros((1, 32, 3), dtype="float32")).shape
    print f_output(np.zeros((1, 32, 3), dtype="float32"))

    # Test passed!


if __name__ == '__main__':
    test_tf_gru_network()
