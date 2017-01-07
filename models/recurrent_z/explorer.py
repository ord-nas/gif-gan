from bottle import route, run, template, static_file, request
from model import DCGAN
import argparse
import tensorflow as tf
import numpy as np
import os
import time
from utils import inverse_transform
import cv2

parser = argparse.ArgumentParser()
# Input/output params
parser.add_argument("--checkpoint_directory", required=True, help="Directory to load checkpoint files from")
parser.add_argument("--tmp_directory", default="/tmp/DCGAN_server_dir", help="Directory to dump temp files")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use internally to generate images")
parser.add_argument("--num_directions", type=int, default=16, help="How many different z-space directions to show")
parser.add_argument("--num_steps", type=int, default=4, help="How many steps to take in each direction")
parser.add_argument("--initial_face_rows", type=int, default=8, help="How many rows of faces when choosing initial face")
parser.add_argument("--initial_face_cols", type=int, default=8, help="How many cols of faces when choosing initial face")
parser.add_argument("--image_size", type=int, default=64, help="Size of images used")
parser.add_argument("--output_size", type=int, default=64, help="Size of output images")
parser.add_argument("--c_dim", type=int, default=3, help="Dimension of image colour")
parser.add_argument("--port", type=int, default=8080, help="Port to use")

class ServerState(object):
    def __init__(self, sess, args, dcgan):
        self.sess = sess
        self.args = args
        self.dcgan = dcgan
        self.video_zs = [] # list of 1d numpy array
        self.video_paths = [] # list of strings
        self.directions = None # 2d numpy array
        self.direction_zs = None # 3d numpy array, direction_zs[direction,step,z_dim]
        self.direction_paths = [] # list of list of strings, direction_paths[direction][step]
        self.add_individually = False
        self.counter = 0
        self.response = None

def write_img(im, tmp_dir, state):
    im = inverse_transform(im)
    im = np.around(im * 255).astype('uint8')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    filename = "img_%d.png" % state.counter
    cv2.imwrite(os.path.join(tmp_dir, filename), im)
    state.counter += 1
    return os.path.join("media/", filename)

# Format of response:
# response: either "success" or "error"
# msg: either an error message or a dictionary of:
#   video_zs: string of textified numpy array
#   video_paths: 1d list of strings (paths to images)
#   directions: string of textified numpy array
#   direction_paths: 2d list of strings (paths to images)

@route('/test_last')
def test_last():
    global state
    if not state.response:
        return {
            "response": "error",
            "msg": "No last response value!",
        }
    return state.response

@route('/test_success')
def test_success():
    global state
    sess = state.sess
    dcgan = state.dcgan
    args = state.args
    num_frames = np.random.randint(4,25)
    video_zs = np.random.uniform(-1.0, 1.0, size=(num_frames, dcgan.z_dim))
    video_imgs = run_inference(sess, dcgan, video_zs)
    video_paths = [write_img(im, args.tmp_directory, state) for im in video_imgs]
    num_directions = np.random.randint(4,17)
    num_steps = np.random.randint(4,17)
    directions = np.random.uniform(-0.1, 0.1, size=(num_directions, dcgan.z_dim))
    direction_zs = np.random.uniform(-1.0, 1.0, size=(num_directions * num_steps, dcgan.z_dim))
    direction_imgs = run_inference(sess, dcgan, direction_zs)
    direction_paths = np.array([write_img(im, args.tmp_directory, state) for im in direction_imgs])
    direction_paths = np.reshape(direction_paths, (num_directions, num_steps)).tolist()
    state.response = {
        "response": "success",
        "msg": {
            "video_zs": repr(video_zs),
            "video_paths": video_paths,
            "directions": repr(direction_zs),
            "direction_paths": direction_paths,
        }
    }
    return state.response

@route('/test_error')
def test_error():
    return {
        "response": "error",
        "msg": "This is an emulation of an error",
    }

@route('/test/<n>')
def test(n):
    global state
    start = time.time()
    sess = state.sess
    dcgan = state.dcgan
    n = int(n)
    imgs = run_inference(sess, dcgan, np.random.uniform(-1.0, 1.0, size=(n, dcgan.z_dim)))
    end = time.time()
    return template("<p>Got {{x}} images in {{t}} seconds!</p>", x=len(imgs), t=end-start)

def update_direction_paths(state):
    (rows, cols, z_dim) = state.direction_zs.shape
    zs = np.reshape(state.direction_zs, [rows * cols, z_dim])
    imgs = run_inference(state.sess, state.dcgan, zs)
    paths = np.array([write_img(im, state.args.tmp_directory, state) for im in imgs])
    state.direction_paths = np.reshape(paths, [rows, cols]).tolist()

def update_direction_imgs(state, step_size):
    if state.directions is None or not state.video_zs:
        return
    assert state.directions.shape == (state.args.num_directions, state.dcgan.z_dim)
    step_size = float(step_size)
    last_z = state.video_zs[-1]
    zs = np.array([[last_z] * state.args.num_steps] * state.args.num_directions)
    for d in xrange(state.args.num_directions):
        for s in xrange(state.args.num_steps):
            zs[d][s] += state.directions[d] * step_size * (s+1)
    zs = np.maximum(-1.0, np.minimum(1.0, zs))
    state.direction_zs = zs
    state.add_individually = False
    update_direction_paths(state)

@route('/init_face')
def init_face():
    global state
    step_size = request.params.get('step_size')
    state.video_zs = [np.random.uniform(-1.0, 1.0, size=(state.dcgan.z_dim,))]
    imgs = run_inference(state.sess, state.dcgan, state.video_zs)
    state.video_paths = [write_img(imgs[0], state.args.tmp_directory, state)]
    update_direction_imgs(state, step_size)
    return get_response(state)

@route('/init_directions')
def init_directions():
    global state
    step_size = request.params.get('step_size')
    directions = np.random.uniform(-1.0, 1.0, size=(state.args.num_directions,
                                                    state.dcgan.z_dim))
    norms = np.sqrt(np.sum(np.square(directions), axis=1, keepdims=True))
    state.directions = np.divide(directions, norms)
    update_direction_imgs(state, step_size)
    return get_response(state)

@route('/clear_directions')
def clear_directions():
    global state
    state.directions = None
    state.direction_zs = None
    state.direction_paths = []
    return get_response(state)

@route('/update_step_size')
def update_step_size():
    global state
    step_size = request.params.get('step_size')
    update_direction_imgs(state, step_size)
    return get_response(state)

@route('/choose_init_face')
def choose_init_face():
    global state
    state.video_zs = []
    state.video_paths = []
    state.directions = None
    state.direction_zs = np.random.uniform(-1.0, 1.0, size=(state.args.initial_face_rows,
                                                            state.args.initial_face_cols,
                                                            state.dcgan.z_dim))
    state.add_individually = True
    update_direction_paths(state)
    return get_response(state)

@route('/add_image')
def add_image():
    global state
    row = int(request.params.get('row'))
    col = int(request.params.get('col'))
    step_size = request.params.get('step_size')
    cols = [col] if state.add_individually else range(col+1)
    for c in cols:
        state.video_zs.append(state.direction_zs[row,c,:])
        state.video_paths.append(state.direction_paths[row][c])
    update_direction_imgs(state, step_size)
    return get_response(state)

@route('/delete_image')
def delete_image():
    global state
    index = int(request.params.get('index'))
    step_size = request.params.get('step_size')
    last = (index == len(state.video_zs) - 1)
    if index >= 0 and index < len(state.video_zs):
        state.video_zs.pop(index)
        state.video_paths.pop(index)
    if last:
        update_direction_imgs(state, step_size)
    return get_response(state)

def get_response(state):
    return {
        "response": "success",
        "msg": {
            "video_zs": repr(state.video_zs),
            "video_paths": state.video_paths,
            "directions": repr(state.directions),
            "direction_paths": state.direction_paths,
        }
    }

@route('/index.html')
def index():
    return static_file('index.html', root='explorer_static')

@route('/blank.jpg')
def index():
    return static_file('blank.jpg', root='explorer_static')

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

    # Configure numpy to print full arrays
    np.set_printoptions(threshold=np.nan)

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
