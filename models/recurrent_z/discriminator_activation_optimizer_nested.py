import cv2
import random
import argparse
import os
import sample_frames
import tensorflow as tf
import numpy as np
from utils import transform, inverse_transform, save_images, get_images
from model import DCGAN
from z_model_lib import VID_DCGAN

def parse_pair(s):
    pair = [int(value) for value in s.split(",")]
    assert len(pair) == 2
    return pair

# Params for algorithm
parser = argparse.ArgumentParser()
parser.add_argument("--input_videos", default=[], nargs='*', help="Search for first frame of these videos")
parser.add_argument("--input_images", default=[], nargs='*', help="Search for these images")
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--num_rows", type=int, default=8)
parser.add_argument("--num_cols", type=int, default=8)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate of for adam [0.0002]")
parser.add_argument("--beta1", type=float, default=0.5, help="Momentum term of adam [0.5]")
parser.add_argument("--discriminator_mode", required=True, choices=["train", "inference"])
parser.add_argument("--sample_dir", required=True, help="Directory name to save the image samples")
parser.add_argument("--reps", type=int, default=1, help="Number of times to repreat each frame")
parser.add_argument("--video_scale", type=int, default=1, help="How much to scale up each dim of video")
parser.add_argument("--sample_frequency", type=int, default=100, help="How often to write samples to disk. Zero for no samples")
# Loss weights
parser.add_argument("--pixel_L2_weight", type=float, default=0.0, help="L2 loss over pixel data")
parser.add_argument("--pixel_L1_weight", type=float, default=0.0, help="L1 loss over pixel data")
parser.add_argument("--activations_L2_weight", type=float, default=1.0, help="L2 loss over discriminator activations")
parser.add_argument("--activations_L1_weight", type=float, default=0.0, help="L1 loss over discriminator activations")
parser.add_argument("--generator_loss_weight", type=float, default=0.0, help="Generator loss weight")
parser.add_argument("--lr_decay_frequency", type=int, default=0, help="How often to decay lr")
parser.add_argument("--lr_decay_amount", type=float, default=0.9, help="Amount to decay lr by")
# DCGAN params
parser.add_argument("--checkpoint_directory", required=True, help="Directory to load checkpoint files from")
parser.add_argument("--image_size", type=int, default=64, help="Size of images used")
parser.add_argument("--output_size", type=int, default=64, help="Size of output images")
parser.add_argument("--c_dim", type=int, default=3, help="Dimension of image colour")
parser.add_argument("--vid_length", type=int, default=16, help="Length of output images")
# GUI
parser.add_argument("--gui", dest="gui", action="store_true")
parser.add_argument("--no_gui", dest="gui", action="store_false")
parser.set_defaults(gui=False)
# Progress video
parser.add_argument("--progress_vid", dest="progress_vid", action="store_true")
parser.add_argument("--no_progress_vid", dest="progress_vid", action="store_false")
parser.set_defaults(progress_vid=False)
parser.add_argument("--progress_vid_sections", default=[], nargs='*', type=parse_pair)
parser.add_argument("--progress_vid_frame_rate", default=10.0, type=float)

def load_dcgan(sess, args):
    batch_size = args.num_rows * args.num_cols
    z = tf.get_variable('z', [batch_size, 120], initializer=tf.random_uniform_initializer(
        minval=-1.0, maxval=1.0))
    sess.run(tf.initialize_all_variables())
    dcgan = VID_DCGAN(sess,
                      batch_size=batch_size,
                      z_input_size=120,
                      z_output_size=100,
                      vid_length=16,
                      input_image_size=args.image_size,
                      output_image_size=args.output_size,
                      c_dim=args.c_dim,
                      sample_cols=args.num_cols,
                      z=z)
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_directory)
    assert ckpt and ckpt.model_checkpoint_path
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver = tf.train.Saver([v for v in tf.all_variables() if v is not z])
    print "RESTORING FROM:", os.path.join(args.checkpoint_directory, ckpt_name)
    saver.restore(sess, os.path.join(args.checkpoint_directory, ckpt_name))
    return dcgan

def load_first_frame(video_file, image_size):
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

def load_image(image_file, image_size):
    frame = cv2.imread(image_file)
    frame_size = (image_size, image_size)
    frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame, is_crop=False)
    return frame

def parse_video_description(path, dcgan):
    name, ext = os.path.splitext(path)
    if ext == ".txt":
        with open(path, 'r') as f:
            description = f.read()
        from numpy import array
        obj = eval(description) # Shut up, this doesn't need to be secure.
    else:
        assert ext == ".npy", "Can't read video description!"
        obj = np.load(path)
        obj = list(obj)
    for x in obj:
        if x.shape != (dcgan.z_dim,):
            raise Exception("z-dim doesn't match")
    return obj

def should_save_frame(i, sections):
    s = sections[0]
    for (index, freq) in sections[1:]:
        if index <= i:
            s = [index, freq]
    return (i - s[0]) % s[1] == 0

def main():
    args = parser.parse_args()
    sess = tf.Session()
    random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    with tf.variable_scope('video_gan'):
        dcgan = load_dcgan(sess, args)

    # Make sample directory if it doesn't exist
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    # Load the list of video files
    targets = []
    for v in args.input_videos:
        targets.append(load_first_frame(v, args.image_size))
    for img in args.input_images:
        targets.append(load_image(img, args.image_size))
    assert len(targets) > 0
    assert dcgan.batch_size % len(targets) == 0
    replicas = dcgan.batch_size / len(targets)
    print "TARGETS:", len(targets), targets[0].shape
    print "REPLICAS:", replicas
    targets_array = np.zeros((dcgan.batch_size, args.vid_length, args.image_size, args.image_size, args.c_dim))
    for (i, img) in enumerate(targets * replicas):
        targets_array[i, 0, :, :, :] = img
    targets_array = np.reshape(targets_array, [-1, args.image_size, args.image_size, args.c_dim])
    print "TARGETS ARRAY:", targets_array.shape
    train = (args.discriminator_mode == "train")
    target_activations_tensor = dcgan.img_dcgan.D_activations if train else dcgan.img_dcgan.D_activations_inf
    target_activations = sess.run(target_activations_tensor, feed_dict={
        dcgan.img_dcgan.images: targets_array,
    })

    # Save the target to disk
    save_images(targets_array[::args.vid_length], [args.num_rows, args.num_cols],
                os.path.join(args.sample_dir, "target.png"))

    # Build optimizers for making the image match the target

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

    # Activations L2 loss
    activations_tensor = dcgan.img_dcgan.D_activations_ if train else dcgan.img_dcgan.D_activations_inf_
    print "activations:", activations_tensor.get_shape().as_list()
    activations_L2 = tf.reduce_mean(tf.square(activations_tensor[::args.vid_length] -
                                              tf.constant(target_activations[::args.vid_length])),
                                    reduction_indices=[1,2,3])
    print "activations L2:", activations_L2.get_shape().as_list()
    activations_L2_loss = tf.reduce_mean(activations_L2)
    # Activations L1 loss
    activations_L1 = tf.reduce_mean(tf.abs(activations_tensor[::args.vid_length] -
                                           tf.constant(target_activations[::args.vid_length])),
                                    reduction_indices=[1,2,3])
    print "activations L1:", activations_L1.get_shape().as_list()
    activations_L1_loss = tf.reduce_mean(activations_L1)
    # Generator loss
    generator_loss = dcgan.img_dcgan.g_loss if train else dcgan.img_dcgan.g_loss_inf
    print "Generator loss:", generator_loss.get_shape().as_list()
    # Pixel L2 loss
    image_tensor = dcgan.img_dcgan.G if train else dcgan.img_dcgan.sampler
    print "Generated image tensor:", image_tensor.get_shape().as_list()
    print "Target image:", targets_array.shape
    pixel_difference = image_tensor[::args.vid_length] - targets_array[::args.vid_length]
    print "Difference:", pixel_difference.get_shape().as_list()
    pixel_L2 = tf.reduce_mean(tf.square(pixel_difference),
                              reduction_indices=[1,2,3])
    print "pixel L2:", pixel_L2.get_shape().as_list()
    pixel_L2_loss = tf.reduce_mean(pixel_L2)
    # Pixel L1 loss
    pixel_L1 = tf.reduce_mean(tf.abs(pixel_difference),
                              reduction_indices=[1,2,3])
    print "pixel L1:", pixel_L1.get_shape().as_list()
    pixel_L1_loss = tf.reduce_mean(pixel_L1)

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
        print "SCOPE VARS", [v.name for v in scope_vars]
        sess.run(tf.initialize_variables(scope_vars))

    imgs_tensor = dcgan.img_dcgan.G if train else dcgan.img_dcgan.sampler

    # Prep a writer if we're making a progress vid
    progress_vid_frame_size = (args.num_cols * args.image_size,
                               args.num_rows * args.image_size)
    progress_vid_sections = args.progress_vid_sections or [[0,1]]
    progress_vid_sections[0][0] = 0
    if args.progress_vid:
        w = cv2.VideoWriter(os.path.join(args.sample_dir, "progress.mp4"),
                            0x20, args.progress_vid_frame_rate, progress_vid_frame_size)
    
    # Actually do the train loop
    current_lr = args.learning_rate
    freq = args.lr_decay_frequency
    for i in xrange(args.num_steps):
        _, loss_value, samples = sess.run([optim, loss, imgs_tensor], feed_dict={
            lr_tensor: current_lr,
            dcgan.is_training: train,
        })
        samples = samples[::args.vid_length]
        if args.gui:
            frame = get_images(samples, [args.num_rows, args.num_cols])
            frame = np.around(frame * 255).astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            height, width, _ = frame.shape
            scale_factor = 4
            frame = cv2.resize(frame, (width * scale_factor, height * scale_factor))
            cv2.imshow("Output", frame)
            key = cv2.waitKey(0)
            while key in [ord('+'), ord('-')]:
                if key == ord('+'):
                    current_lr /= args.lr_decay_amount
                    print "SET LEARING RATE TO:", current_lr
                else:
                    current_lr *= args.lr_decay_amount
                    print "SET LEARING RATE TO:", current_lr
                key = cv2.waitKey(0)
            if key == ord('q'):
                break
        # Write samples to disk, if applicable
        if args.sample_frequency > 0 and i % args.sample_frequency == 0:
            save_images(samples, [args.num_rows, args.num_cols],
                        os.path.join(args.sample_dir, "train_%d.png" % i))
            print "Saved sample"
        # Decay learning rate, if applicable
        if freq > 0 and i % freq == freq - 1:
            current_lr *= args.lr_decay_amount
            print "SET LEARING RATE TO:", current_lr
        # Update progress video, if applicable
        write_progress_frame = args.progress_vid and should_save_frame(i, progress_vid_sections)
        if write_progress_frame:
            frame = get_images(samples, [args.num_rows, args.num_cols])
            assert frame.shape == (progress_vid_frame_size[1], progress_vid_frame_size[0], 3)
            frame = np.around(frame * 255).astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            w.write(frame)
        print "Step %d/%d: loss %f%s" % (i, args.num_steps, loss_value,
                                         " [saved progress frame]" if write_progress_frame else "")
            
    # Finish off progress vid
    if args.progress_vid:
        w.release()

    samples = sess.run(imgs_tensor, feed_dict={dcgan.is_training:train})
    samples = samples[::args.vid_length]
    save_images(samples, [args.num_rows, args.num_cols],
                os.path.join(args.sample_dir, "final.png"))
    print "Saved final images"

    sz = args.output_size
    vid_samples = sess.run(imgs_tensor, feed_dict={
        dcgan.is_training: train,
    })
    videos = np.reshape(vid_samples, [args.num_rows,
                                      args.num_cols,
                                      args.vid_length,
                                      sz,
                                      sz,
                                      args.c_dim])

    # Write the clean samples
    filename = os.path.join(args.sample_dir, "final.mp4")
    print "Writing samples to", filename
    w = cv2.VideoWriter(filename,
                        0x20,
                        25.0,
                        (args.num_cols * sz, args.num_rows * sz))
    for t in xrange(args.vid_length):
        frame = np.zeros(shape=[args.num_rows * sz,
                                args.num_cols * sz,
                                args.c_dim],
                         dtype=np.uint8)
        for r in xrange(args.num_rows):
            for c in xrange(args.num_cols):
                im = inverse_transform(videos[r,c,t,:,:,:])
                im = np.around(im * 255).astype('uint8')
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                frame[r*sz:(r+1)*sz, c*sz:(c+1)*sz, :] = im
        w.write(frame)
    w.release()
    

if __name__ == "__main__":
    main()
