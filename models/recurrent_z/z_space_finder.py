import cv2
import random
import argparse
import os
import sample_frames
import tensorflow as tf
import numpy as np
from utils import transform, save_images, get_images, inverse_transform
from model import DCGAN

# Params for algorithm
parser = argparse.ArgumentParser()
# Basic IO
parser.add_argument("--video_list", required=True, nargs='+', help="List(s) of videos to use")
parser.add_argument("--video_dataset_dir", required=True, help="Directory to read dataset from")
parser.add_argument("--output_z_folder", required=True, help="Directory to write z-vectors")
parser.add_argument("--output_comparison_folder", default="", help="Optional directory to write side-by-side comparison videos")
parser.add_argument("--output_image_folder", default="", help="Optional directory to write final images")
parser.add_argument("--output_frame_folder", default="", help="Optional directory to write frame-by-frame images")
# General params
parser.add_argument("--video_batch_size", type=int, default=8, help="How many videos to process at once")
parser.add_argument("--stop_after", type=int, default=0, help="Debug option; bail after n batches")
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--num_initial_steps", type=int, default=500)
parser.add_argument("--num_steps_per_frame", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate of for adam [0.0002]")
parser.add_argument("--lr_decay_amount", type=float, default=0.5, help="Amount to decay lr by")
parser.add_argument("--beta1", type=float, default=0.5, help="Momentum term of adam [0.5]")
parser.add_argument("--discriminator_mode", required=True, choices=["train", "inference"])
parser.add_argument("--vid_length", type=int, default=16, help="Number of frames to use")
parser.add_argument("--frame_skip", type=int, default=2, help="Frame step size")
# Loss weights
parser.add_argument("--pixel_L2_weight", type=float, default=0.0, help="L2 loss over pixel data")
parser.add_argument("--pixel_L1_weight", type=float, default=0.0, help="L1 loss over pixel data")
parser.add_argument("--activations_L2_weight", type=float, default=1.0, help="L2 loss over discriminator activations")
parser.add_argument("--activations_L1_weight", type=float, default=0.0, help="L1 loss over discriminator activations")
parser.add_argument("--generator_loss_weight", type=float, default=0.0, help="Generator loss weight")
# DCGAN params
parser.add_argument("--checkpoint_directory", required=True, help="Directory to load checkpoint files from")
parser.add_argument("--image_size", type=int, default=64, help="Size of images used")
parser.add_argument("--output_size", type=int, default=64, help="Size of output images")
parser.add_argument("--c_dim", type=int, default=3, help="Dimension of image colour")

def load_dcgan(sess, args):
    batch_size = args.video_batch_size
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

def load_video(video_file, image_size, vid_length, skip):
    cap = cv2.VideoCapture(video_file)
    frames = []
    for _ in xrange(vid_length):
        for _ in xrange(skip):
            if not cap.isOpened():
                print "Video %s not long enough! Skipping!" % video_file
                return None
            ret, im = cap.read()
            if not ret:
                print "Video %s not long enough! Skipping!" % video_file
                return None
        frame_size = (image_size, image_size)
        frame = cv2.resize(im, frame_size, interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame, is_crop=False)
        frames.append(frame)
    return frames

def fpath(fname):
    return os.path.join(args.video_dataset_dir, fname)

def outpath(fname, base=None, ext=".npy"):
    if not base:
        base = args.output_z_folder
    basename = os.path.basename(fname)
    (basename, _) = os.path.splitext(basename)
    return os.path.join(base, "%s%s" % (basename, ext))

def prep_for_mp4(im, scaled_size):
    im = inverse_transform(im)
    im = np.around(im * 255).astype('uint8')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    im = cv2.resize(im, scaled_size, interpolation=cv2.INTER_LINEAR)
    return im

def process_batch(
        batch,
        batch_fname,
        dcgan,
        optim,
        loss,
        target_activations_tensor,
        imgs_tensor,
        lr_tensor,
        activations_placeholder,
        target_placeholder,
        sess):

    # Construct numpy array for the image batch and activation batch
    targets = np.array(batch)
    full_shape = targets.shape
    all_target_activations = np.stack([sess.run(target_activations_tensor,
                                                feed_dict={
                                                    dcgan.images: targets[:, i, :, :, :],
                                                }) for i in xrange(args.vid_length)],
                                      axis=1)

    # Actually do the train loop
    num_steps = args.num_initial_steps + args.num_steps_per_frame * args.vid_length
    full_results = np.zeros(full_shape)
    full_z = np.zeros([args.video_batch_size, args.vid_length, dcgan.z_dim])
    current_lr = args.learning_rate
    for i in xrange(args.num_initial_steps):
        _, loss_value = sess.run([optim, loss], feed_dict={
            lr_tensor: current_lr,
            activations_placeholder: all_target_activations[:, 0, :],
            target_placeholder: targets[:, 0, :, :, :],
        })
        print "Step %d/%d: loss %f" % (i, num_steps, loss_value)
    full_results[:, 0, :, :, :] = sess.run(imgs_tensor)
    full_z[:, 0, :] = sess.run(dcgan.z)
    current_lr *= args.lr_decay_amount
    for f in xrange(args.vid_length):
        for i in xrange(args.num_steps_per_frame):
            global_step = args.num_initial_steps + args.num_steps_per_frame * f + i
            _, loss_value = sess.run([optim, loss], feed_dict={
                lr_tensor: current_lr,
                activations_placeholder: all_target_activations[:, f, :],
                target_placeholder: targets[:, f, :, :, :],
            })
            print "Step %d/%d: loss %f" % (global_step, num_steps, loss_value)
        full_results[:, f, :, :, :] = sess.run(imgs_tensor)
        full_z[:, f, :] = sess.run(dcgan.z)

    print "Writing output for this batch ..."
        
    # Write final images to file
    if args.output_image_folder:
        for (result, fname) in zip(full_results, batch_fname):
            fpath = outpath(fname, base=args.output_image_folder, ext=".png")
            save_images(result, [1, args.vid_length], fpath)
            print "Wrote final images to %s" % fpath

    # Write final frames to file
    if args.output_frame_folder:
        for (result, fname) in zip(full_results, batch_fname):
            folder = outpath(fname, base=args.output_frame_folder, ext="")
            if not os.path.exists(folder):
                os.makedirs(folder)
            for i in xrange(args.vid_length):
                save_images(result[i:i+1,:,:,:], [1, 1],
                            os.path.join(folder, "%03d.png" % i))
            print "Wrote final frames to %s" % folder

    # Write comparison video to file
    if args.output_comparison_folder:
        scale_factor = 2
        scaled_size = (scale_factor * args.image_size, scale_factor * args.image_size)
        height, width = scaled_size[0], 2*scaled_size[1]
        for (target, result, fname) in zip(targets, full_results, batch_fname):
            fpath = outpath(fname, base=args.output_comparison_folder, ext=".mp4")
            w = cv2.VideoWriter(fpath, 0x20, 25.0, (width, height))
            for t in xrange(args.vid_length):
                im_real = prep_for_mp4(target[t,:,:,:], scaled_size)
                im_result = prep_for_mp4(result[t,:,:,:], scaled_size)
                w.write(np.concatenate([im_real, im_result], axis=1))
            w.release()

    # Write the actual z data
    for (z_result, fname) in zip(full_z, batch_fname):
        np.save(outpath(fname), z_result)
            
    print "Done output for this batch"

def main():
    # Initialization
    global args
    args = parser.parse_args()
    sess = tf.Session()
    random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    dcgan = load_dcgan(sess, args)

    # Make output directories if they don't exist
    if not os.path.exists(args.output_z_folder):
        os.makedirs(args.output_z_folder)
    if args.output_comparison_folder and not os.path.exists(args.output_comparison_folder):
        os.makedirs(args.output_comparison_folder)
    if args.output_image_folder and not os.path.exists(args.output_image_folder):
        os.makedirs(args.output_image_folder)
    if args.output_frame_folder and not os.path.exists(args.output_frame_folder):
        os.makedirs(args.output_frame_folder)
        
    # Load the list of video files
    files = []
    for lst in args.video_list:
        with open(lst, 'r') as f:
            for video in f:
                video = video.strip()
                if not video:
                    continue
                files.append(video)
    print "Total video files found:", len(files)

    # Normalize the weights
    loss_weights = ['pixel_L2_weight',
                    'pixel_L1_weight',
                    'activations_L2_weight',
                    'activations_L1_weight',
                    'generator_loss_weight']
    total = sum([getattr(args, lw) for lw in loss_weights])
    print "Normalized loss weights:"
    for lw in loss_weights:
        setattr(args, lw, getattr(args, lw) / total)
        print lw, getattr(args, lw)

    # Find key nodes in graph
    train = (args.discriminator_mode == "train")
    target_activations_tensor = dcgan.D_activations if train else dcgan.D_activations_inf
    activations_tensor = dcgan.D_activations_ if train else dcgan.D_activations_inf_
    imgs_tensor = dcgan.G if train else dcgan.sampler

    # Make placeholders for targets
    target_placeholder = tf.placeholder(
        tf.float32, dcgan.sampler.get_shape().as_list())
    activations_placeholder = tf.placeholder(
        tf.float32, target_activations_tensor.get_shape().as_list())

    # Define losses

    # Activations L2 loss
    print "activations:", activations_tensor.get_shape().as_list()
    activations_L2 = tf.reduce_mean(tf.square(activations_tensor - activations_placeholder),
                                    reduction_indices=[1,2,3])
    print "activations L2:", activations_L2.get_shape().as_list()
    activations_L2_loss = tf.reduce_mean(activations_L2)
    # Activations L1 loss
    activations_L1 = tf.reduce_mean(tf.abs(activations_tensor - activations_placeholder),
                                    reduction_indices=[1,2,3])
    print "activations L1:", activations_L1.get_shape().as_list()
    activations_L1_loss = tf.reduce_mean(activations_L1)
    # Generator loss
    generator_loss = dcgan.g_loss if train else dcgan.g_loss_inf
    print "generator loss:", generator_loss.get_shape().as_list()
    # Pixel L2 loss
    image_tensor = dcgan.G if train else dcgan.sampler
    print "generated image tensor:", image_tensor.get_shape().as_list()
    pixel_difference = image_tensor - target_placeholder
    print "difference:", pixel_difference.get_shape().as_list()
    pixel_L2 = tf.reduce_mean(tf.square(pixel_difference),
                              reduction_indices=[1,2,3])
    print "pixel L2:", pixel_L2.get_shape().as_list()
    pixel_L2_loss = tf.reduce_mean(pixel_L2)
    # Pixel L1 loss
    pixel_L1 = tf.reduce_mean(tf.abs(pixel_difference),
                              reduction_indices=[1,2,3])
    print "pixel L1:", pixel_L1.get_shape().as_list()
    pixel_L1_loss = tf.reduce_mean(pixel_L1)
    
    # Build optimizer for making the image match the target
    with tf.variable_scope("optimizers"):
        # Compute overall loss
        loss = (activations_L2_loss * args.activations_L2_weight +
                activations_L1_loss * args.activations_L1_weight +
                pixel_L2_loss * args.pixel_L2_weight +
                pixel_L1_loss * args.pixel_L1_weight +
                generator_loss * args.generator_loss_weight)
        lr_tensor = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(lr_tensor, beta1=args.beta1)
        optim = optimizer.minimize(loss, var_list=[dcgan.z])
        scope_vars = tf.get_collection(tf.GraphKeys.VARIABLES,
                                       scope=tf.get_variable_scope().name)
        sess.run(tf.initialize_variables(scope_vars))

    # Iterate over all the videos
    i = 0
    batch_count = 0
    while i < len(files):
        if args.stop_after > 0 and batch_count >= args.stop_after:
            break
        batch = []
        batch_fname = []
        batch_count += 1
        while len(batch) < args.video_batch_size and i < len(files):
            fname = files[i]
            i += 1
            if os.path.exists(outpath(fname)):
                print "Skipping %s because already processed" % fname
                continue
            vid = load_video(fpath(fname), args.image_size, args.vid_length, args.frame_skip)
            if vid:
                batch.append(vid)
                batch_fname.append(fname)
        # If we have less than one batch left over at the end, we just throw it
        # out because I'm lazy
        if len(batch) == args.video_batch_size:
            process_batch(batch,
                          batch_fname,
                          dcgan,
                          optim,
                          loss,
                          target_activations_tensor,
                          imgs_tensor,
                          lr_tensor,
                          activations_placeholder,
                          target_placeholder,
                          sess)
        else:
            for fname in batch_fname:
                print "Skipping %s because not enough to make a full batch!" % fname

if __name__ == "__main__":
    main()
