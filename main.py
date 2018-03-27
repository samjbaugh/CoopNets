from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import tensorflow as tf

from model.model import CoopNets
from util.util import make_dir, init_tf_log

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer('image_size', 64, 'Image size to rescale images')
tf.flags.DEFINE_integer('batch_size', 100, 'Batch size of training images')
tf.flags.DEFINE_integer('num_epochs', 250, 'Number of epochs to train')
tf.flags.DEFINE_integer('n_tile_row', 12, 'Row number of synthesized images')
tf.flags.DEFINE_integer('n_tile_col', 12, 'Column number of synthesized images')
tf.flags.DEFINE_float('beta1', 0.5, 'Momentum term of adam')

# parameters for descriptor
tf.flags.DEFINE_float('d_lr', 0.01, 'Initial learning rate for descriptor')
tf.flags.DEFINE_float('des_refsig', 0.016, 'Standard deviation for reference distribution of descriptor')
tf.flags.DEFINE_integer('des_sample_steps', 10, 'Sample steps for Langevin dynamics of descriptor')
tf.flags.DEFINE_float('des_step_size', 0.002, 'Step size for descriptor Langevin dynamics')

# parameters for generator
tf.flags.DEFINE_float('g_lr', 0.0001, 'Initial learning rate for generator')
tf.flags.DEFINE_float('gen_refsig', 0.3, 'Standard deviation for reference distribution of generator')
tf.flags.DEFINE_integer('gen_sample_steps', 0, 'Sample steps for Langevin dynamics of generator')
tf.flags.DEFINE_float('gen_step_size', 0.1, 'Step size for generator Langevin dynamics')

tf.flags.DEFINE_string('data_dir', './data', 'The data directory')
tf.flags.DEFINE_string('category', 'rock', 'The name of dataset')
tf.flags.DEFINE_string('output_dir', './output', 'The output directory for saving results')
tf.flags.DEFINE_integer('log_step', 10, 'Number of epochs to save output results')
tf.flags.DEFINE_boolean('test', False, 'True if in testing mode')
tf.flags.DEFINE_string('ckpt', None, 'Checkpoint path to load')
tf.flags.DEFINE_integer('sample_size', 144, 'Number of images to generate during test.')


def main(_):
    run_name = datetime.now().strftime('%Y:%m:%d:%H:%M:%S')

    output_dir = os.path.join(FLAGS.output_dir, FLAGS.category, run_name)
    sample_dir = make_dir(os.path.join(output_dir, 'synthesis'))
    log_dir = make_dir(os.path.join(output_dir, 'log'))
    model_dir = make_dir(os.path.join(output_dir, 'checkpoints'))
    test_dir = make_dir(os.path.join(output_dir, 'test'))

    init_tf_log(os.path.join(output_dir, 'log', 'output.log'))

    model = CoopNets(
        num_epochs=FLAGS.num_epochs,
        image_size=FLAGS.image_size,
        batch_size=FLAGS.batch_size,
        beta1=FLAGS.beta1,
        n_tile_row=FLAGS.n_tile_row, n_tile_col=FLAGS.n_tile_col,
        d_lr=FLAGS.d_lr, g_lr=FLAGS.g_lr,
        des_refsig=FLAGS.des_refsig, gen_refsig=FLAGS.gen_refsig,
        des_step_size=FLAGS.des_step_size, gen_step_size=FLAGS.gen_step_size,
        des_sample_steps=FLAGS.des_sample_steps, gen_sample_steps=FLAGS.gen_sample_steps,
        log_step=FLAGS.log_step, data_path=FLAGS.data_dir, category=FLAGS.category,
        sample_dir=sample_dir, log_dir=log_dir, model_dir=model_dir, test_dir=test_dir
    )

    with tf.Session() as sess:
        if FLAGS.test:
            model.test(sess, FLAGS.ckpt, FLAGS.sample_size)
        else:
            model.train(sess)


if __name__ == '__main__':
    tf.app.run()
