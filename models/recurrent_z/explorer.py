"""
Little webserver app for playing around with image DCGAN latent z space.

Run like this:

(in this directory)

python explorer.py --checkpoint_directory <path/to/checkpoint/directory> --save_directory <path/to/directory/to/dump/saved/videos> --batch_size 1 --port 8080

For example:

python explorer.py --checkpoint_directory DCGAN_checkpoints/r0/face_stills_v0_64_64_saver_version_1/ --save_directory MY_SAVE_DIR --batch_size 1 --port 8080
"""

from bottle import route, run, template, static_file, request, BaseRequest
from model import DCGAN
import argparse
import tensorflow as tf
import numpy as np
import os
import time
from utils import inverse_transform
import cv2
import sys


BaseRequest.MEMFILE_MAX = 1024 * 1024

parser = argparse.ArgumentParser()
# Input/output params
parser.add_argument("--checkpoint_directory", required=True, help="Directory to load checkpoint files from")
parser.add_argument("--tmp_directory", default="/tmp/DCGAN_server_dir", help="Directory to dump temp files")
parser.add_argument("--save_directory", required=True, help="Directory to place save files")
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
        self.vid_counter = 0
        self.response = None
        self.video_filename = "None"

# Returns the *client-side* path to the image
def write_img(im, state):
    im = inverse_transform(im)
    im = np.around(im * 255).astype('uint8')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    filename = "img_%d.png" % state.counter
    state.counter += 1
    cv2.imwrite(os.path.join(state.args.tmp_directory, filename), im)
    return os.path.join("media/", filename)

# Returns the *actual* path to the video
def write_video(frame_rate, state):
    filename = "vid_%d.mp4" % state.vid_counter
    filepath = os.path.join(state.args.save_directory, filename)
    state.vid_counter += 1
    frame_size = (state.args.image_size*2, state.args.image_size*2)
    writer = cv2.VideoWriter(filepath, 0x20, frame_rate, frame_size)
    imgs = run_inference(state.sess, state.dcgan, state.video_zs)
    for im in imgs:
        im = inverse_transform(im)
        im = np.around(im * 255).astype('uint8')
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, frame_size, interpolation=cv2.INTER_LINEAR)
        writer.write(im)
    writer.release()
    return filepath

# Format of response:
# response: either "success" or "error"
# msg: either an error message or a dictionary of:
#   video_zs: string of textified numpy array
#   video_paths: 1d list of strings (paths to images)
#   directions: string of textified numpy array
#   direction_paths: 2d list of strings (paths to images)

@route('/test_last', method=['GET', 'POST'])
def test_last():
    global state
    if not state.response:
        return {
            "response": "error",
            "msg": "No last response value!",
        }
    return state.response

@route('/test_success', method=['GET', 'POST'])
def test_success():
    global state
    sess = state.sess
    dcgan = state.dcgan
    args = state.args
    num_frames = np.random.randint(4,25)
    video_zs = np.random.uniform(-1.0, 1.0, size=(num_frames, dcgan.z_dim))
    video_imgs = run_inference(sess, dcgan, video_zs)
    video_paths = [write_img(im, state) for im in video_imgs]
    num_directions = np.random.randint(4,17)
    num_steps = np.random.randint(4,17)
    directions = np.random.uniform(-0.1, 0.1, size=(num_directions, dcgan.z_dim))
    direction_zs = np.random.uniform(-1.0, 1.0, size=(num_directions * num_steps, dcgan.z_dim))
    direction_imgs = run_inference(sess, dcgan, direction_zs)
    direction_paths = np.array([write_img(im, state) for im in direction_imgs])
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

@route('/test_error', method=['GET', 'POST'])
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
    paths = np.array([write_img(im, state) for im in imgs])
    state.direction_paths = np.reshape(paths, [rows, cols]).tolist()

def update_direction_imgs(state, step_size):
    if state.directions is None or not state.video_zs:
        return
    assert state.directions.shape == (state.args.num_directions, state.dcgan.z_dim)
    step_size = step_size
    last_z = state.video_zs[-1]
    zs = np.array([[last_z] * state.args.num_steps] * state.args.num_directions)
    for d in xrange(state.args.num_directions):
        for s in xrange(state.args.num_steps):
            zs[d][s] += state.directions[d] * step_size * (s+1)
    #zs = np.maximum(-1.0, np.minimum(1.0, zs))
    state.direction_zs = zs
    state.add_individually = False
    update_direction_paths(state)

@route('/init_face', method=['GET', 'POST'])
def init_face():
    global state
    step_size = float(request.params.get('step_size'))
    state.video_zs = [np.random.uniform(-1.0, 1.0, size=(state.dcgan.z_dim,))]
    imgs = run_inference(state.sess, state.dcgan, state.video_zs)
    state.video_paths = [write_img(imgs[0], state)]
    update_direction_imgs(state, step_size)
    return get_response(state)

@route('/init_directions', method=['GET', 'POST'])
def init_directions():
    global state
    step_size = float(request.params.get('step_size'))
    directions = np.random.uniform(-1.0, 1.0, size=(state.args.num_directions,
                                                    state.dcgan.z_dim))
    norms = np.sqrt(np.sum(np.square(directions), axis=1, keepdims=True))
    state.directions = np.divide(directions, norms)
    update_direction_imgs(state, step_size)
    return get_response(state)

@route('/clear_directions', method=['GET', 'POST'])
def clear_directions():
    global state
    state.directions = None
    state.direction_zs = None
    state.direction_paths = []
    return get_response(state)

@route('/update_step_size', method=['GET', 'POST'])
def update_step_size():
    global state
    step_size = float(request.params.get('step_size'))
    update_direction_imgs(state, step_size)
    return get_response(state)

@route('/random_faces', method=['GET', 'POST'])
def random_faces():
    global state
    state.directions = None
    state.direction_zs = np.random.uniform(-1.0, 1.0, size=(state.args.initial_face_rows,
                                                            state.args.initial_face_cols,
                                                            state.dcgan.z_dim))
    state.add_individually = True
    update_direction_paths(state)
    return get_response(state)

@route('/clear_faces', method=['GET', 'POST'])
def clear_faces():
    global state
    state.video_zs = []
    state.video_paths = []
    return get_response(state)

@route('/perp_faces', method=['GET', 'POST'])
def perp_faces():
    global state
    similarity = float(request.params.get('similarity'))
    if len(state.video_zs) < 2:
        return get_error("Need at least two faces in timeline to get perpendicular faces")
    first = state.video_zs[0]
    last = state.video_zs[-1]
    delta = last - first
    max_index = np.argmax(np.abs(delta))
    delta_without_max = np.concatenate((delta[:max_index], delta[max_index+1:]))
    max_val = delta[max_index]
    perp_seeds = np.random.uniform(-1.0, 1.0, size=(state.args.initial_face_rows,
                                                    state.args.initial_face_cols,
                                                    state.dcgan.z_dim-1))
    inferred_value = -1.0 / max_val * np.sum(np.multiply(perp_seeds,
                                                         delta_without_max),
                                             axis=2, keepdims=True)
    perp = np.concatenate((perp_seeds[:,:,:max_index],
                           inferred_value,
                           perp_seeds[:,:,max_index:]),
                        axis=2)
    norms = np.sqrt(np.sum(np.square(perp), axis=2, keepdims=True))
    perp = np.divide(perp, norms) * similarity
    state.direction_zs = first + perp
    state.add_individually = True
    #state.direction_zs = np.maximum(-1.0, np.minimum(1.0, state.direction_zs))
    update_direction_paths(state)
    return get_response(state)

@route('/add_image', method=['GET', 'POST'])
def add_image():
    global state
    row = int(request.params.get('row'))
    col = int(request.params.get('col'))
    step_size = float(request.params.get('step_size'))
    cols = [col] if state.add_individually else range(col+1)
    for c in cols:
        state.video_zs.append(state.direction_zs[row,c,:])
        state.video_paths.append(state.direction_paths[row][c])
    update_direction_imgs(state, step_size)
    return get_response(state)

@route('/get_similar', method=['GET', 'POST'])
def get_similar():
    global state
    row = int(request.params.get('row'))
    col = int(request.params.get('col'))
    step_size = float(request.params.get('step_size'))
    similarity = float(request.params.get('similarity'))
    if state.add_individually:
        initial = state.direction_zs[row,col,:]
        deltas = np.random.uniform(-1.0, 1.0, size=(state.args.initial_face_rows,
                                                    state.args.initial_face_cols,
                                                    state.dcgan.z_dim))
        norms = np.sqrt(np.sum(np.square(deltas), axis=1, keepdims=True))
        deltas = np.divide(deltas, norms) * similarity
        deltas[0,0,:] = 0.0 # Top left should be the initial faces
        state.direction_zs = initial + deltas
        #state.direction_zs = np.maximum(-1.0, np.minimum(1.0, state.direction_zs))
        update_direction_paths(state)
        return get_response(state)
    else:
        initial = state.directions[row,:]
        deltas = np.random.uniform(-1.0, 1.0, size=(state.args.num_directions,
                                                    state.dcgan.z_dim))
        norms = np.sqrt(np.sum(np.square(deltas), axis=1, keepdims=True))
        deltas = np.divide(deltas, norms) * similarity
        deltas[0,:] = 0.0 # Top should be identical faces
        directions = initial + deltas
        direction_norms = np.sqrt(np.sum(np.square(directions), axis=1, keepdims=True))
        state.directions = np.divide(directions, direction_norms)
        update_direction_imgs(state, step_size)
        return get_response(state)

@route('/delete_image', method=['GET', 'POST'])
def delete_image():
    global state
    index = int(request.params.get('index'))
    step_size = float(request.params.get('step_size'))
    last = (index == len(state.video_zs) - 1)
    if index >= 0 and index < len(state.video_zs):
        state.video_zs.pop(index)
        state.video_paths.pop(index)
    if last:
        update_direction_imgs(state, step_size)
    return get_response(state)

def parse_video_description(description, state):
    from numpy import array
    obj = eval(description) # Shut up, this doesn't need to be secure.
    for x in obj:
        if x.shape != (state.dcgan.z_dim,):
            raise Exception("z-dim doesn't match")
        # if np.max(x) > 1.0 or np.min(x) < -1.0:
        #     raise Exception("value(s) out of range")
    return obj

@route('/load_video_description', method=['GET', 'POST'])
def load_video_description():
    global state
    description = request.params.get('description')
    step_size = float(request.params.get('step_size'))
    try:
        state.video_zs = parse_video_description(description, state)
        imgs = run_inference(state.sess, state.dcgan, state.video_zs)
        state.video_paths = [write_img(im, state) for im in imgs]
        update_direction_imgs(state, step_size)
        return get_response(state)
    except:
        e = sys.exc_info()[1]
        return get_error("Problem parsing video description: %s" % e)

@route('/load_relative_video_description', method=['GET', 'POST'])
def load_relative_video_description():
    global state
    if not state.video_zs:
        return get_error("Need at least one existing video frame to load relative")
    description = request.params.get('description')
    step_size = float(request.params.get('step_size'))
    try:
        abs_d = parse_video_description(description, state)
        rel_d = [np.subtract(x, abs_d[0]) for x in abs_d]
        last = state.video_zs[-1]
        new = [np.add(x, last) for x in rel_d[1:]]
        #new = np.maximum(-1.0, np.minimum(1.0, new))
        state.video_zs.extend(new)
        imgs = run_inference(state.sess, state.dcgan, new)
        state.video_paths.extend([write_img(im, state) for im in imgs])
        update_direction_imgs(state, step_size)
        return get_response(state)
    except:
        e = sys.exc_info()[1]
        return get_error("Problem parsing video description: %s" % e)

def get_error(msg):
    return {
        "response": "error",
        "msg": msg,
    }
    
def get_response(state):
    return {
        "response": "success",
        "msg": {
            "video_zs": repr(state.video_zs),
            "video_paths": state.video_paths,
            "directions": repr(state.directions),
            "direction_paths": state.direction_paths,
            "video_save_path": state.video_filename,
        }
    }

@route('/index.html')
def index():
    return static_file('index.html', root='explorer_static')

@route('/blank.jpg')
def blank():
    return static_file('blank.jpg', root='explorer_static')

@route('/save', method=['GET', 'POST'])
def save():
    global state
    try:
        frame_rate = int(request.params.get('frame_rate'))
    except ValueError:
        return get_error("Couldn't parse FPS")
    state.video_filename = write_video(frame_rate, state)
    return get_response(state)

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

    # Make directories if they don't exist
    if not os.path.exists(args.tmp_directory):
        os.makedirs(args.tmp_directory)
    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)

    # Define route for tmp directory
    @route('/media/<filename>')
    def media(filename):
        return static_file(filename, root=args.tmp_directory)

    run(host='localhost', port=args.port)

if __name__ == "__main__":
    main()
