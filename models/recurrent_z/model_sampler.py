import tensorflow as tf
import numpy as np
import os

import utils
from z_model_lib import VID_DCGAN

flags = tf.app.flags
# Model params
flags.DEFINE_integer("vid_batch_size", 64, "The size of batch videos [64]")
flags.DEFINE_integer("vid_length", 16, "The length of the videos [16]")
flags.DEFINE_integer("image_size", 64, "The size of images used [64]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_float("image_noise", 0.0, "Std of noise to add to images")
flags.DEFINE_float("activation_noise", 0.0, "Std of noise to add to D activations")
# "Actual" args
flags.DEFINE_string("checkpoint_dir", "", "Directory to load checkpoint from")
flags.DEFINE_integer("num_samples", 1000, "Number of sample gifs to generate [1000]")
flags.DEFINE_string("output_directory", "", "Directory to write output gifs")
flags.DEFINE_integer("random_seed", 0, "Random numpy seed to use [0]")
FLAGS = flags.FLAGS

def main(_):
    utils.pp.pprint(flags.FLAGS.__flags)

    np.random.seed(FLAGS.random_seed)

    with tf.Session() as sess:
        with tf.variable_scope('video_gan'):
            vid_z_dim = 120
            image_z_dim = 100

            # Build the model
            vid_dcgan = VID_DCGAN(sess,
                                  FLAGS.vid_batch_size,
                                  vid_z_dim,
                                  image_z_dim,
                                  FLAGS.vid_length,
                                  FLAGS.image_size,
                                  FLAGS.output_size,
                                  c_dim=FLAGS.c_dim,
                                  image_noise_std=FLAGS.image_noise,
                                  activation_noise_std=FLAGS.activation_noise)

            # Load model weights.
            vid_dcgan.load_checkpoint(sess, FLAGS.checkpoint_dir)

            fps = 25.0
            duration = (1.0 / fps) * vid_dcgan.vid_length

            for i in xrange(0, FLAGS.num_samples, vid_dcgan.batch_size):
                # Generate some z-vectors for one video.
                sample_z = np.random.uniform(-1, 1, size=(vid_dcgan.batch_size, vid_z_dim))
                samples = sess.run(vid_dcgan.img_dcgan.sampler, feed_dict={
                    vid_dcgan.z: sample_z,
                    vid_dcgan.is_training: False,
                })
                videos = np.reshape(samples, [vid_dcgan.batch_size,
                                              vid_dcgan.vid_length,
                                              vid_dcgan.output_image_size,
                                              vid_dcgan.output_image_size,
                                              vid_dcgan.c_dim])
                upper_bound = min(FLAGS.num_samples, i+vid_dcgan.batch_size)
                print "Writing samples %d to %d ..." % (i, upper_bound)
                for j in xrange(i, upper_bound):
                    filename = os.path.join(FLAGS.output_directory, "%d.gif" % j)
                    utils.make_gif(videos[j - i, :, :, :, :], filename, duration)

if __name__ == '__main__':
    tf.app.run()

