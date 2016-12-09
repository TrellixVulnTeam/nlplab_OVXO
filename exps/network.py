from sandbox.rocky.tf.core.network import MLP, GRUNetwork
import tensorflow as tf
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.parameterized import Model
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from exps.layers import AttnGRULayer, GRULayer
from sandbox.rocky.tf.misc import tensor_utils
import numpy as np


class RewardMLP(MLP):
    """
    overrides MLP with methods / properties used in generative adversarial learning.
    """

    def compute_reward(self, X):
        predits = -tf.log(1.0 - self.output)
        # predits = -tf.log(1.0 - tf.sigmoid(self.output))
        Y_p = self._predict(predits, X)
        return Y_p

    def compute_score(self, X):
        """
        predict logits ...
        """
        logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
        # logits = self.output
        Y_p = self._predict(logits, X)
        return Y_p

    def likelihood_loss(self):
        logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
        # logits = L.get_output(self.layers[-1])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.target_var)
        # ent_B = tfutil.logit_bernoulli_entropy(logits)
        # self.obj = tf.reduce_sum(loss_B - self.ent_reg_weight * ent_B)
        return tf.reduce_sum(loss)

    def complexity_loss(self, reg, cmx):
        return tf.constant(0.0)

    def loss(self, reg=0.0, cmx=0.0):
        # logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.target_var)
        # ent_B = tfutil.logit_bernoulli_entropy(logits)
        # self.obj = tf.reduce_sum(loss_B - self.ent_reg_weight * ent_B)
        # return tf.reduce_sum(loss)
        loss = self.likelihood_loss()
        return loss


class RewardGRUNetwork(GRUNetwork, Model, LayersPowered):
#class RewardGRUNetwork(GRUNetwork, Model):
    """
    Hmmmm it's just a normal network,
    but with regression (affine transform) in the end

    The input dim is vocab_size (action_space) + hidden_dim (observation_space/char_dim)

    GRUNetwork does compute logits..so no need to add MLP on top
    """

    # TODO 1: allow attention with critic
    # TODO 2: disable attention on critic (focus on language modeling) (pretrain with LM task???)
    # TODO    for now, to be simple, we don't want attention (but it can be easily added)
    # TODO 3: test this? see if it can be properly trained (after knowing what rewards are)

    # TODO 1: Figure out what paper means by inputting "correct" label
    # TODO 2: Have Alex add weight regularization to this network!

    def __init__(self, name, config,
                 learning_rate=0.0003,
                 max_gradient_norm=10.0,
                 gru_layer_cls=GRULayer,
                 hidden_dim=None, **kwargs):

        self.save_name = "reward_gru_network"

        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm

        action_dim = config["vocab_size"]

        input_dim = config["vocab_size"] + config["gru_size"]

        l_input = L.InputLayer(
            shape=(None, None, input_dim),
            name="input"
        )
        if hidden_dim is None:
            hidden_dim = config["gru_size"]

        # output_non_linearity is None, so it will be tf.identity
        # we are not using softmax or other nonlinearity
        layer_args = {'regularizer': tf.nn.l2_loss}
        super(RewardGRUNetwork, self).__init__(name=name,
                                               input_shape=(input_dim,),
                                               hidden_dim=hidden_dim,
                                               output_dim=action_dim,
                                               input_layer=l_input,
                                               output_nonlinearity=tf.identity,  # this is already the default option
                                               gru_layer_cls=gru_layer_cls,
                                               layer_args=layer_args,
                                               **kwargs)

        self.f_output = tensor_utils.compile_function(
            inputs=[
                self.input_layer.input_var  # (batch_size, time_step, input_dim)
            ],
            outputs=L.get_output(
                self.output_layer
            )
        )

        # problem with tensorflow, must use tf.float32, even though the data is int32
        self.y_label = tf.placeholder(tf.float32, shape=[None, None])  # batch_size, time_step

        # also we need action input of the time step to calculate loss
        self.action_input = tf.placeholder(tf.float32, shape=[None, None, action_dim]) # should be one-hot encoding

        self.global_step = tf.Variable(0, trainable=False)  # hmmm from NLC
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.setup_loss()

        # don't know how well this works...at least I can claim it runs...
        LayersPowered.__init__(self, self.output_layer, input_layers=[l_input])

    def compute_reward(self, X):
        """
        Parameters
        ----------
        X: (batch_size, max_seq_len, input_dim) - this is put in from processed sampled_data
        X: (max_seq_len, vocab_size + gru_size / input_dim)
            So effectively this is not the shape (batch_size, time_step, input_dim)
            solution - e.xpand X to (1, max_seq_len, input_dim)
            (even though Alex doesn't agree with this shape, we assume this is the right one for now)
        Returns
        --------
        a reward vector of shape (max_seq_len,)
        """
        if len(X.shape) == 2:  # make sure there's no batch
            X = np.expand_dims(X, axis=0)
        rewards = self.f_output(X)  # output_layer is a DenseLayer in GRUNetwork
        rewards = np.squeeze(rewards)
        return rewards

    def compute_reward_sa(self, obs_var, actions_var):
        """

        Parameters
        ----------
        obs_var: numpy arrays!!!!!!!!!
        actions_var

        They are not the same
        even though obs is action, this obs is the previous timestep
        action is for the future timestep

        We can consider using word embedding of the action,
        instead of a one-hot vector

        Returns
        -------
        """
        X = np.concatenate((obs_var, actions_var),axis=2)
        rewards = self.f_output(X)  # output_layer is a DenseLayer in GRUNetwork
        rewards = np.squeeze(rewards)
        return rewards

    def setup_loss(self):
        # construct symbolic output
        output = L.get_output(self.output_layer)

        # doshape = tf.shape(output)
        # batch_size, T = doshape[0], doshape[1]  # output from GRUNetwork is reshaped!!!! so it's normal...

        output2d = tf.squeeze(output)

        # squared loss, batch_size 1
        # weight penalty is hard-coded by Alex
        reg = 0.1
        reg_losses = filter(lambda x: x.name.split('/')[0] == 'reward_gru', tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        mean_reg_loss = reg * tf.reduce_mean(reg_losses) # / tf.to_float(batch_size)

        # self.action_input: [batch_size, seq_len, action_dim]
        # output2d: [batch_size, seq_len, action_dim]

        Q_action = tf.reduce_sum(tf.mul(output2d, self.action_input), reduction_indices=2)
        self.loss = tf.reduce_mean(tf.square(self.y_label - Q_action)) + mean_reg_loss

        # borrowed from NLC code, we do grad-clipping

        params = tf.trainable_variables()  # TODO: eh, THIS MIGHT NOT WORK :( considering how many models we have

        # ERROR: tensorflow.python.framework.errors.InvalidArgumentError: Incompatible shapes: [2,32,52] vs. [2,32]

        gradients = tf.gradients(self.loss, params)  # hopefully this picks out params that are really used
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        #      self.gradient_norm = tf.global_norm(clipped_gradients)
        self.gradient_norm = tf.global_norm(gradients)
        self.param_norm = tf.global_norm(params)
        self.updates = self.optimizer.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

    def train(self, obs_var, actions_var, y_label):
        """
        Instead of using Solver or FirstOrderOptimizer
        we optimize this network normally

        we only need input, and y_label (reward ground truth)
        also, pass in the session (maybe a different session?)
        """

        # we get default session
        session = tf.get_default_session()

        X = np.concatenate((obs_var, actions_var),axis=2)
        assert len(X.shape) == 3

        input_feed = {}
        input_feed[self.input_layer.input_var] = X
        input_feed[self.y_label] = y_label
        input_feed[self.action_input] = actions_var

        output_feed = [self.updates, self.loss, self.gradient_norm, self.param_norm]

        outputs = session.run(output_feed, input_feed)
        return outputs[1]  # return loss

if __name__ == '__main__':
    from exps.nlc_env import build_data
    
    # path_2_ptb_data = "/home/alex/stanford_dev/sisl/rllab/ptb_data/"
    path_2_ptb_data = "/Users/Aimingnie/Documents/School/Stanford/AA228/nlplab/ptb_data/"

    #x_train = "/Users/Aimingnie/Documents" + "/School/Stanford/AA228/nlplab/ptb_data/train.ids.x"
    #y_train = "/Users/Aimingnie/Documents" + "/School/Stanford/AA228/nlplab/ptb_data/train.ids.y"

    #x_dev = "/Users/Aimingnie/Documents" + "/School/Stanford/AA228/nlplab/ptb_data/valid.ids.x"
    #y_dev = "/Users/Aimingnie/Documents" + "/School/Stanford/AA228/nlplab/ptb_data/valid.ids.y"

    x_train = "{}/train.ids.x".format(path_2_ptb_data)
    y_train = "{}/train.ids.y".format(path_2_ptb_data)

    x_dev = "{}/valid.ids.x".format(path_2_ptb_data)
    y_dev = "{}/valid.ids.y".format(path_2_ptb_data)

    vocab_path = "{}/vocab.dat".format(path_2_ptb_data)

    source_tokens, source_mask, target_tokens, target_mask = build_data(fnamex="{}/train.ids.x".format(path_2_ptb_data),
                                                                        fnamey="{}/train.ids.y".format(path_2_ptb_data),
                                                                        num_layers=1, max_seq_len=200)   

    from rl_train import create_model
    sess = tf.Session()
    model = create_model(sess, 52, False)
    L_dec = model.L_dec.eval(session=sess)
    L_enc = model.L_enc.eval(session=sess)     
    config = {
        "gru_size": 32,  # remember train.py for NLC must be adjusted as well
        "source": source_tokens,
        "target": target_tokens,
        "source_mask": source_mask,
        "target_mask": target_mask,
        "model": model,
        "L_dec": L_dec,  # vocab_size: 52, hidden_size: 50
        "L_enc": L_enc,
        "measure": "CER",
        'data_dir': path_2_ptb_data,
        "max_seq_len": 32,
        # This is determined by data preprocessing (for decoder it's 32, cause <start>, <end>)
        "batch_size": 64,  # batch_size must be multiples of max_seq_len (did you mix
        # batch-size with n_env in policy code???)
        "sess": sess,  # tf_session
        "vocab_size": L_dec.shape[0],
        "encoder_max_seq_length": 30
    }
    critic = RewardGRUNetwork("reward_gru", config)
    
    halt= True
