from rllab.misc import ext
from rllab.misc import logger
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.misc import tensor_utils
# from rllab.algo.first_order_method import parse_update_method
from rllab.optimizers.minibatch_dataset import BatchDataset
from collections import OrderedDict
import tensorflow as tf
import time
from functools import partial
import pyprind


class FirstOrderOptimizer(Serializable):
    """
    Performs (stochastic) gradient descent, possibly using fancier methods like adam etc.
    """

    def __init__(
            self,
            tf_optimizer_cls=None,
            tf_optimizer_args=None,
            # learning_rate=1e-3,
            max_epochs=1000,
            tolerance=1e-6,
            batch_size=32,
            callback=None,
            verbose=False,
            **kwargs):
        """
        :param max_epochs:
        :param tolerance:
        :param update_method:
        :param batch_size: None or an integer. If None the whole dataset will be used.
        :param callback:
        :param kwargs:
        :return:
        """
        Serializable.quick_init(self, locals())
        self._opt_fun = None
        self._target = None
        self._callback = callback
        if tf_optimizer_cls is None:
            tf_optimizer_cls = tf.train.AdamOptimizer
        if tf_optimizer_args is None:
            tf_optimizer_args = dict(learning_rate=1e-3)
        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._verbose = verbose
        self._input_vars = None
        self._train_op = None

    def update_opt(self, loss, target, inputs, extra_inputs=None, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """

        self._target = target

        self._train_op = self._tf_optimizer.minimize(loss, var_list=target.get_params(trainable=True))

        # updates = OrderedDict([(k, v.astype(k.dtype)) for k, v in updates.iteritems()])

        if extra_inputs is None:
            extra_inputs = list()
        self._input_vars = inputs + extra_inputs
        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(inputs + extra_inputs, loss),
        )

    def loss(self, inputs, extra_inputs=None):
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_loss"](*(tuple(inputs) + extra_inputs))

    def optimize(self, inputs, extra_inputs=None, callback=None):

        if len(inputs) == 0:
            # Assumes that we should always sample mini-batches
            raise NotImplementedError

        f_loss = self._opt_fun["f_loss"]

        if extra_inputs is None:
            extra_inputs = tuple()

        last_loss = f_loss(*(tuple(inputs) + extra_inputs))

        start_time = time.time()

        dataset = BatchDataset(inputs, self._batch_size, extra_inputs=extra_inputs)

        sess = tf.get_default_session()

        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % (epoch))
                progbar = pyprind.ProgBar(len(inputs[0]))

            for batch in dataset.iterate(update=True):
                sess.run(self._train_op, dict(list(zip(self._input_vars, batch))))
                if self._verbose:
                    progbar.update(len(batch[0]))

            if self._verbose:
                if progbar.active:
                    progbar.stop()

            new_loss = f_loss(*(tuple(inputs) + extra_inputs))

            if self._verbose:
                logger.log("Epoch: %d | Loss: %f" % (epoch, new_loss))
            if self._callback or callback:
                elapsed = time.time() - start_time
                callback_args = dict(
                    loss=new_loss,
                    params=self._target.get_param_values(trainable=True) if self._target else None,
                    itr=epoch,
                    elapsed=elapsed,
                )
                if self._callback:
                    self._callback(callback_args)
                if callback:
                    callback(**callback_args)

            if abs(last_loss - new_loss) < self._tolerance:
                break
            last_loss = new_loss

"""
optimizer = FOO(max_epochs= args.epochs, batch_size=args.batch_size, tolerance= 1e-6)
optimizer.update_opt(model.loss(reg= args.reg, cmx= args.cmx), model, [model.input_var], extra_inputs= [model.target_var],
                     like_loss= model.likelihood_loss(),
                     cmpx_loss= model.complexity_loss(args.reg, args.cmx))
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    optimizer.optimize([X_t], extra_inputs=tuple([Y_t]), callback= viz.append_data,
                 val_inputs= [X_v], val_extra_inputs= tuple([Y_v]))
"""

class Solver(object):
    """
    Convenience class wrapping the first order optimizer
    """
    def __init__(self, model, reg, cmx, max_epochs, batch_size, tolerance,
                 tf_optimizer_cls= None, tf_optimizer_args= None, callback= None):
        self._optimizer = FirstOrderOptimizer(max_epochs= max_epochs, batch_size= batch_size, tolerance= tolerance,
                                              tf_optimizer_cls= tf_optimizer_cls, tf_optimizer_args= tf_optimizer_args)
        self._optimizer.update_opt(model.loss(reg= reg, cmx= cmx), model, [model.input_var], extra_inputs= [model.target_var],
                     like_loss= model.likelihood_loss(), cmpx_loss= model.complexity_loss(reg, cmx))
        self._callback = callback

    def train(self, X_train, Y_train, X_validate= None, Y_validate= None):
        self._optimizer.optimize([X_train], extra_inputs= tuple([Y_train]), callback= self._callback,
                                val_inputs= [X_validate], val_extra_inputs= tuple([Y_validate]))
        
        
class SimpleSolver(object):
    def __init__(self, model, epochs, batch_size):
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name='lr')
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.opt = self.optimizer.minimize(model.loss())

    def train(self, X_train, Y_train, lr):
        sess = tf.get_default_session()

        N = X_train.shape[0]
        for epoch in range(self.epochs):
            p = np.random.permutation(N)
            X_train = X_train[p]
            Y_train = Y_train[p]

            for i in range(0,N,self.batch_size):
                X_batch = X_train[i:i+self.batch_size]
                Y_batch = Y_train[i:i+self.batch_size]

                sess.run([self.opt],{self.model.input_var:X_batch,
                                     self.model.target_var:Y_batch,
                                     self.learning_rate:lr})
