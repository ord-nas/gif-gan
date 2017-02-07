# TODO: this borrows a lot from discriminator_activation_optimizer.py, should
# really be refactored
# TODO: should try the methodology from the image completion paper, which is
# optimizing L2 loss between target and generation, with an added term for
# discriminator score.

import cv2
import random
import argparse
import os
import sample_frames
import tensorflow as tf
import numpy as np
from utils import transform, save_images, inverse_transform
from model import DCGAN

# Params for algorithm
parser = argparse.ArgumentParser()
parser.add_argument("--input_video", required=True, help="Search for first frame of this video")
parser.add_argument("--input_path", required=True, help="Path to apply")
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--num_rows", type=int, default=8)
parser.add_argument("--num_cols", type=int, default=8)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate of for adam [0.0002]")
parser.add_argument("--beta1", type=float, default=0.5, help="Momentum term of adam [0.5]")
parser.add_argument("--discriminator_mode", required=True, choices=["train", "inference"])
parser.add_argument("--sample_dir", required=True, help="Directory name to save the image samples")
parser.add_argument("--reps", type=int, default=1, help="How many times to repeat each video frame")
# DCGAN params
parser.add_argument("--checkpoint_directory", required=True, help="Directory to load checkpoint files from")
parser.add_argument("--image_size", type=int, default=64, help="Size of images used")
parser.add_argument("--output_size", type=int, default=64, help="Size of output images")
parser.add_argument("--c_dim", type=int, default=3, help="Dimension of image colour")

def load_dcgan(sess, args):
    batch_size = args.num_rows * args.num_cols
    z = tf.get_variable('z', [batch_size, 100], initializer=tf.random_uniform_initializer(
        minval=-1.0, maxval=1.0))
    sess.run(tf.initialize_all_variables())
    dcgan = DCGAN(sess,
                  image_size=args.image_size,
                  batch_size=(batch_size),
                  output_size=args.output_size,
                  c_dim=args.c_dim,
                  dataset_name='',
                  is_crop=False,
                  checkpoint_dir='',
                  sample_dir='',
                  data_dir='',
                  log_dir='',
                  image_glob='',
                  shuffle=False,
                  z=z)
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_directory)
    assert ckpt and ckpt.model_checkpoint_path
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver = tf.train.Saver([v for v in tf.all_variables() if v is not z])
    saver.restore(sess, os.path.join(args.checkpoint_directory, ckpt_name))
    return dcgan

def load_image(video_file, image_size):
    cap = cv2.VideoCapture(video_file)
    frame = None
    if cap.isOpened():
        ret, im = cap.read()
        if ret:
            frame = im
    assert frame is not None
    frame_size = (image_size, image_size)
    frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame, is_crop=False)
    return frame

def parse_video_description(description, dcgan):
    from numpy import array
    obj = eval(description) # Shut up, this doesn't need to be secure.
    for x in obj:
        if x.shape != (dcgan.z_dim,):
            raise Exception("z-dim doesn't match")
    return obj

def main():
    args = parser.parse_args()
    sess = tf.Session()
    random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    dcgan = load_dcgan(sess, args)

    # Make sample directory if it doesn't exist
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    # Load the path description
    with open(args.input_path, 'r') as f:
        description = f.read()
    abs_d = parse_video_description(description, dcgan)
    rel_d = [np.subtract(x, abs_d[0]) for x in abs_d]

    # Load the list of video files
    target = load_image(args.input_video, args.image_size)
    train = (args.discriminator_mode == "train")
    target_activations_tensor = dcgan.D_activations if train else dcgan.D_activations_inf
    target_activations = sess.run(target_activations_tensor, feed_dict={
        dcgan.images: np.array([target] * dcgan.batch_size),
    })
    print "TARGET ACTIVATIONS:", target_activations[0]

    # Save the target to disk
    save_images(np.array([target]), [1, 1],
                os.path.join(args.sample_dir, "target.png"))

    # Build optimizers for making the images' activations match the target
    activations_tensor = dcgan.D_activations_ if train else dcgan.D_activations_inf_
    print activations_tensor.get_shape().as_list()
    distance = tf.sqrt(tf.reduce_sum(tf.square(activations_tensor - tf.constant(target_activations)),
                                     reduction_indices=[1,2,3]))
    print distance.get_shape().as_list()
    loss = tf.reduce_mean(distance)
    print loss.get_shape().as_list()
    with tf.variable_scope("optimizers"):
        lr = tf.placeholder(tf.float32, [], name='learning_rate')
        optimizer = tf.train.AdamOptimizer(lr, beta1=args.beta1)
        optim = optimizer.minimize(loss, var_list=[dcgan.z])
        scope_vars = tf.get_collection(tf.GraphKeys.VARIABLES,
                                       scope=tf.get_variable_scope().name)
        print "SCOPE VARS", [v.name for v in scope_vars]
        sess.run(tf.initialize_variables(scope_vars))

    imgs_tensor = dcgan.G if train else dcgan.sampler

    # Actually do the train loop
    learning_rate = args.learning_rate
    for i in xrange(args.num_steps):
        _, loss_value = sess.run([optim, loss], feed_dict={
            lr: learning_rate,
        })
        print "Step %d/%d: loss %f" % (i, args.num_steps, loss_value)
        if i % 100 == 0:
            samples = sess.run(imgs_tensor)
            save_images(samples, [args.num_rows, args.num_cols],
                        os.path.join(args.sample_dir, "train_%d.png" % i))
            print sess.run(dcgan.z)
            print "Saved sample"
        if i % 750 == 0 and i != 0:
            learning_rate = learning_rate * 0.5

    samples = sess.run(imgs_tensor)
    save_images(samples, [args.num_rows, args.num_cols],
                os.path.join(args.sample_dir, "final.png"))
    print "Saved final images"

    print "Final distances:"
    ds = sess.run(distance)
    for d in ds:
        print d

    print "Sanity check distances:"
    final_activations = sess.run(target_activations_tensor, feed_dict={
        dcgan.images: samples,
    })
    for i in xrange(dcgan.batch_size):
        a = final_activations[i]
        print np.linalg.norm(a - target_activations[0])

    print "Okay, generating output video."
    initial_z = sess.run(dcgan.z)
    zs = [np.add(x, initial_z) for x in rel_d]
    batches = [sess.run(imgs_tensor, feed_dict={dcgan.z: z}) for z in zs]

    sz = args.image_size
    frame_size = (args.num_cols * sz, args.num_cols * sz)
    w = cv2.VideoWriter(os.path.join(args.sample_dir, "video.mp4"),
                        0x20, 25.0, frame_size)
    for batch in batches:
        frame = np.zeros(shape=[args.num_rows * sz,
                                args.num_rows * sz,
                                args.c_dim],
                         dtype=np.uint8)
        for r in xrange(args.num_rows):
            for c in xrange(args.num_cols):
                im = inverse_transform(batch[r*args.num_cols+c,:,:])
                im = np.around(im * 255).astype('uint8')
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                frame[r*sz:(r+1)*sz, c*sz:(c+1)*sz, :] = im
        for _ in xrange(args.reps):
            w.write(frame)
    w.release()

if __name__ == "__main__":
    main()
