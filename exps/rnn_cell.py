import tensorflow as tf

from tensorflow.python.ops.rnn_cell import *
import tensorflow.python.util.nest as nest

class GRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, input_size=None, activation=tanh):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with vs.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                out, M_gates, W1, W2, b_gates = _linear([inputs, state], 2 * self._num_units, True, 1.0)
                self.W_xr, self.W_xu = tf.split(1,2,W1)
                self.W_hr, self.W_hu = tf.split(1,2,W2)
                
                self.M_gates = M_gates
                self.b_gates = b_gates
                
                r, u = array_ops.split(1, 2, out)
                r, u = sigmoid(r), sigmoid(u)
            with vs.variable_scope("Candidate"):
                out, M_candidates, W1, W2, b_candidate = _linear([inputs, r * state], self._num_units, True)
                self.W_xc, self.W_hc = W1, W2
                
                self.M_candidate = M_candidates
                self.b_candidate = b_candidate
                
                c = self._activation(out)
            new_h = u * state + (1 - u) * c
        return new_h, new_h
    
def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = vs.get_variable(
            "Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable(
            "Bias", [output_size],
            dtype=dtype,
            initializer=init_ops.constant_initializer(
                bias_start, dtype=dtype))
        
    #W1, W2 = tf.split(0, 2, matrix)
    in_dim = shapes[0][-1]
    W1, W2 = matrix[:in_dim], matrix[in_dim:]
    return res + bias_term, matrix, W1, W2, bias_term

if __name__ == '__main__':
    import numpy as np
    
    num_units = 30
    batch_size = 100
    input_size = 50

    cell = GRUCell(num_units, activation=tf.nn.tanh)
    inputs = tf.constant(np.random.randn(batch_size,input_size))
    state = tf.constant(np.random.randn(batch_size,num_units))
    
    cell.__call__(inputs, state)
    
    v = tf.get_collection(tf.GraphKeys.VARIABLES)
    
    halt= True