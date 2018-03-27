import sys
import logging

import tensorflow as tf


def init_log(path):
    log = logging.getLogger()
    log.setLevel(logging.INFO)

    formatter_cs = logging.Formatter('%(message)s')

    cs = logging.StreamHandler(sys.stdout)
    cs.setLevel(logging.INFO)
    cs.setFormatter(formatter_cs)
    log.addHandler(cs)

    log = logging.getLogger('tensorflow')
    log.setLevel(logging.INFO)
    log.handlers = []

    formatter_fh = logging.Formatter('%(asctime)s - %(message)s')

    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_fh)
    log.addHandler(fh)


def make_dir(output_dir):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
    return output_dir
