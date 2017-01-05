from bottle import route, run, template, static_file
from model import DCGAN
import argparse
import tensorflow as tf
import numpy as np
import os

parser = argparse.ArgumentParser()
# Input/output params
parser.add_argument("--checkpoint_directory", required=True, help="Directory to load checkpoint files from")
parser.add_argument("--tmp_directory", default="/tmp/DCGAN_server_dir", help="Directory to dump temp files")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use internally to generate images")
parser.add_argument("--num_directions", type=int, default=16, help="How many different z-space directions to show")
parser.add_argument("--num_steps", type=int, default=4, help="How many steps to take in each direction")
parser.add_argument("--image_size", type=int, default=64, help="Size of images used")
parser.add_argument("--output_size", type=int, default=64, help="Size of output images")
parser.add_argument("--c_dim", type=int, default=3, help="Dimension of image colour")
parser.add_argument("--port", type=int, default=8080, help="Port to use")

class ServerState(object):
    def __init__(self, sess, args, dcgan):
        self.sess = sess
        self.args = args
        self.dcgan = dcgan
        self.video_imgs = []
        self.directions = None
        self.start = None

@route('/test/<n>')
def test(n):
    global state
    sess = state.sess
    dcgan = state.dcgan
    n = int(n)
    imgs = run_inference(sess, dcgan, np.random.uniform(-1.0, 1.0, size=(n, dcgan.z_dim)))
    return template("<p>Got {{x}} images!</p>", x=len(imgs))

@route('/index.html')
def index():
    return static_file('index.html', root='explorer_static')

def load_dcgan(sess, args):
    dcgan = DCGAN(sess,
                  image_size=args.image_size,
                  batch_size=args.batch_size,
                  output_size=args.output_size,
                  c_dim=args.c_dim,
                  dataset_name='',
                  is_crop=False,
                  checkpoint_dir='',
                  sample_dir='',
                  data_dir='',
                  log_dir='',
                  image_glob='',
                  shuffle=False)
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_directory)
    assert ckpt and ckpt.model_checkpoint_path
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    dcgan.saver.restore(sess, os.path.join(args.checkpoint_directory, ckpt_name))
    return dcgan

def run_inference(sess, dcgan, inputs):
    imgs = []
    for i in xrange(0, len(inputs), dcgan.batch_size):
        z_array = inputs[i:i+dcgan.batch_size]
        z = np.stack(z_array)
        z.resize(dcgan.batch_size, dcgan.z_dim)
        samples = sess.run(dcgan.sampler, feed_dict={
            dcgan.z: z,
        })
        imgs.extend(samples[:len(z_array)])
    return imgs

def main():
    global state
    args = parser.parse_args()
    sess = tf.Session()
    dcgan = load_dcgan(sess, args)
    state = ServerState(sess, args, dcgan)

    # Make tmp directory
    if not os.path.exists(args.tmp_directory):
        os.makedirs(args.tmp_directory)

    # Define route for tmp directory
    @route('/media/<filename>')
    def media(filename):
        return static_file(filename, root=args.tmp_directory)

    run(host='localhost', port=args.port)

if __name__ == "__main__":
    main()
