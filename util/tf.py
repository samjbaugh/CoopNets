import tensorflow as tf


def get_lr(adam):
    _beta1_power, _beta2_power = adam._get_beta_accumulators()
    current_lr = (adam._lr_t * tf.sqrt(1 - _beta2_power) / (1 - _beta1_power))
    return current_lr