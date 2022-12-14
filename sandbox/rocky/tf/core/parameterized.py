from contextlib import contextmanager

from rllab.core.serializable import Serializable
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors
import tensorflow as tf
import numpy as np

import h5py
import os
import time

load_params = True

@contextmanager
def suppress_params_loading():
    global load_params
    load_params = False
    yield
    load_params = True


class Parameterized(object):
    def __init__(self):
        self._cached_params = {}
        self._cached_param_dtypes = {}
        self._cached_param_shapes = {}
        self._cached_assign_ops = {}
        self._cached_assign_placeholders = {}
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.save_name = self.save_name + timestr
        print self.save_name

    def get_params_internal(self, **tags):
        """
        Internal method to be implemented which does not perform caching
        """
        raise NotImplementedError

    def get_params(self, **tags):
        """
        Get the list of parameters, filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(**tags)
        return self._cached_params[tag_tuple]

    def get_param_dtypes(self, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_dtypes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_dtypes[tag_tuple] = [val.dtype for val in param_values]
        return self._cached_param_dtypes[tag_tuple]

    def get_param_shapes(self, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_shapes[tag_tuple] = [val.shape for val in param_values]
        return self._cached_param_shapes[tag_tuple]

    def get_param_values(self, **tags):
        params = self.get_params(**tags)
        param_values = tf.get_default_session().run(params)
        return flatten_tensors(param_values)

    def set_param_values(self, flattened_params, **tags):
        debug = tags.pop("debug", False)
        param_values = unflatten_tensors(
            flattened_params, self.get_param_shapes(**tags))
        ops = []
        feed_dict = dict()
        for param, dtype, value in zip(
                self.get_params(**tags),
                self.get_param_dtypes(**tags),
                param_values):
            if param not in self._cached_assign_ops:
                assign_placeholder = tf.placeholder(dtype=param.dtype.base_dtype)
                assign_op = tf.assign(param, assign_placeholder)
                self._cached_assign_ops[param] = assign_op
                self._cached_assign_placeholders[param] = assign_placeholder
            ops.append(self._cached_assign_ops[param])
            feed_dict[self._cached_assign_placeholders[param]] = value.astype(dtype)
            if debug:
                print("setting value of %s" % param.name)
        tf.get_default_session().run(ops, feed_dict=feed_dict)

    def flat_to_params(self, flattened_params, **tags):
        return unflatten_tensors(flattened_params, self.get_param_shapes(**tags))

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        global load_params
        if load_params:
            d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        global load_params
        if load_params:
            tf.get_default_session().run(tf.initialize_variables(self.get_params()))
            self.set_param_values(d["params"])


class JointParameterized(Parameterized):
    def __init__(self, components):
        super(JointParameterized, self).__init__()
        self.components = components

    def get_params_internal(self, **tags):
        params = [param for comp in self.components for param in comp.get_params_internal(**tags)]
        # only return unique parameters
        return sorted(set(params), key=hash)


class Model(Parameterized):
    _load_dir = './weights'
    _log_dir = './weights'

    def load_params(self, filename, itr, skip_params):
        print 'loading policy params...'
        if not hasattr(self, 'load_dir'):
            log_dir = Model._load_dir
        else:
            log_dir = self.load_dir
        filename = log_dir + "/" + filename + '.h5'
        assignments = []

        # create log_dir if non-existent
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with h5py.File(filename,'r') as hf:
            if itr >= 0:
                prefix = self._prefix(itr)
            else:
                prefix = hf.keys()[itr] + "/"

            for param in self.get_params():
                path = prefix + param.name
                if param.name in skip_params:
                    continue

                if path in hf:
                    assignments.append(
                        param.assign(hf[path][...])
                        )
                else:
                    halt= True

        sess = tf.get_default_session()
        sess.run(assignments)
        print 'done.'


    def save_params(self, itr, overwrite= False):
        print 'saving model...'
        if not hasattr(self, 'log_dir'):
            log_dir = Model._log_dir
        else:
            log_dir = self.log_dir
        filename = log_dir + "/" + self.save_name + '.h5'
        sess = tf.get_default_session()

        # create log_dir if non-existent
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        key = self._prefix(itr)
        with h5py.File(filename, 'a') as hf:
            if key in hf:
                dset = hf[key]
            else:
                dset = hf.create_group(key)

            vs = self.get_params()
            vals = sess.run(vs)

            for v, val in zip(vs, vals):
                dset[v.name] = val
        print 'done.'

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir

    def set_load_dir(self, load_dir):
        self.load_dir = load_dir

    @staticmethod
    def _prefix(x):
        return 'iter{:05}/'.format(x)
