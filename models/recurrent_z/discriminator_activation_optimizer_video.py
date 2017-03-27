import cv2
import random
import argparse
import os
import sample_frames
import tensorflow as tf
import numpy as np
from utils import transform, save_images, get_images
from model import DCGAN

# Params for algorithm
parser = argparse.ArgumentParser()
parser.add_argument("--input_videos", required=True, nargs='+', help="Recreate these videos")
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate of for adam [0.0002]")
parser.add_argument("--beta1", type=float, default=0.5, help="Momentum term of adam [0.5]")
parser.add_argument("--discriminator_mode", required=True, choices=["train", "inference"])
parser.add_argument("--sample_dir", required=True, help="Directory name to save the image samples")
parser.add_argument("--vid_length", type=int, default=16, help="Number of frames to use")
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
# More
parser.add_argument("--gui", dest="gui", action="store_true")
parser.add_argument("--no_gui", dest="gui", action="store_false")
parser.set_defaults(gui=False)

def load_dcgan(sess, args):
    batch_size = len(args.input_videos) * args.vid_length
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

def load_video(video_file, image_size, vid_length):
    cap = cv2.VideoCapture(video_file)
    frames = []
    skip = 2
    for _ in xrange(vid_length):
        for _ in xrange(skip):
            assert cap.isOpened(), "Video %s not long enough!" % video_file
            ret, im = cap.read()
            assert ret, "Video %s not long enough!" % video_file
        frame_size = (image_size, image_size)
        frame = cv2.resize(im, frame_size, interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame, is_crop=False)
        frames.append(frame)
    return frames

def main():
    args = parser.parse_args()
    sess = tf.Session()
    random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    dcgan = load_dcgan(sess, args)

    # Make sample directory if it doesn't exist
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    # Load the list of video files
    targets_full = []
    for v in args.input_videos:
        targets_full.append(load_video(v, args.image_size, args.vid_length))
    targets_full = np.array(targets_full)
    full_shape = targets_full.shape
    targets = np.reshape(targets_full, [-1, args.image_size, args.image_size, args.c_dim])
    print "TARGETS:", targets.shape
    train = (args.discriminator_mode == "train")
    target_activations_tensor = dcgan.D_activations if train else dcgan.D_activations_inf
    target_activations = sess.run(target_activations_tensor, feed_dict={
        dcgan.images: targets,
    })
    print "TARGET ACTIVATIONS:", target_activations.shape

    # Save the target to disk
    save_images(targets, [len(args.input_videos), args.vid_length],
                os.path.join(args.sample_dir, "target.png"))

    # Now save some stupid video-esque stuff
    target_frames_folder = os.path.join(args.sample_dir, "target_frames")
    if not os.path.exists(target_frames_folder):
        # No recursive os.makedirs
        os.mkdir(target_frames_folder)
    for i in xrange(args.vid_length):
        save_images(targets_full[:,i,:,:,:], [1, len(args.input_videos)],
                    os.path.join(target_frames_folder, "target_frame_%03d.png" % i))
    
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
    activations_tensor = dcgan.D_activations_ if train else dcgan.D_activations_inf_
    print "activations:", activations_tensor.get_shape().as_list()
    activations_L2 = tf.reduce_mean(tf.square(activations_tensor - tf.constant(target_activations)),
                                    reduction_indices=[1,2,3])
    print "activations L2:", activations_L2.get_shape().as_list()
    activations_L2_loss = tf.reduce_mean(activations_L2)
    # Activations L1 loss
    activations_L1 = tf.reduce_mean(tf.abs(activations_tensor - tf.constant(target_activations)),
                                    reduction_indices=[1,2,3])
    print "activations L1:", activations_L1.get_shape().as_list()
    activations_L1_loss = tf.reduce_mean(activations_L1)
    # Generator loss
    generator_loss = dcgan.g_loss if train else dcgan.g_loss_inf
    print "Generator loss:", generator_loss.get_shape().as_list()
    # Pixel L2 loss
    image_tensor = dcgan.G if train else dcgan.sampler
    print "Generated image tensor:", image_tensor.get_shape().as_list()
    print "Target image:", targets.shape
    pixel_difference = image_tensor - targets
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

    imgs_tensor = dcgan.G if train else dcgan.sampler

    # Actually do the train loop
    current_lr = args.learning_rate
    freq = args.lr_decay_frequency
    should_train = True
    dummy_tensor = tf.constant(1)
    goto = 0
    for i in xrange(args.num_steps):
        opt = optim if should_train else dummy_tensor
        _, loss_value, samples = sess.run([opt, loss, imgs_tensor], feed_dict={
            lr_tensor: current_lr,
        })
        should_train = True
        print "Step %d/%d: loss %f" % (i, args.num_steps, loss_value)
        if args.gui and goto < i:
            frame = get_images(samples, [len(args.input_videos), args.vid_length])
            frame = np.around(frame * 255).astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            height, width, _ = frame.shape
            scale_factor = 2.3
            frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
            cv2.imshow("Output", frame)
            key = cv2.waitKey(0)
            while key in [ord('+'), ord('-')]:
                if key == ord('+'):
                    current_lr /= args.lr_decay_amount
                    print "SET LEARING RATE TO:", current_lr
                else: # key == ord('-')
                    current_lr *= args.lr_decay_amount
                    print "SET LEARING RATE TO:", current_lr
                key = cv2.waitKey(0)
            if key == ord('q'):
                break
            elif key == ord('g'):
                goto = int(raw_input("Goto step: "))
            elif ord('0') <= key <= ord('9'):
                should_train = False
                key_n = key - ord('0')
                if key_n+1 >= args.vid_length:
                    continue
                z = sess.run(dcgan.z)
                z = np.reshape(z, [len(args.input_videos), args.vid_length, dcgan.z_dim])
                z[:, key_n+1:, :] = z[:, key_n:key_n+1, :]
                z = np.reshape(z, [-1, dcgan.z_dim])
                sess.run(dcgan.z.assign(z))
        if i % 100 == 0:
            samples = sess.run(imgs_tensor)
            save_images(samples, [len(args.input_videos), args.vid_length],
                        os.path.join(args.sample_dir, "train_%05d.png" % i))
            print sess.run(dcgan.z)
            print "Saved sample"
        # TODO: TEMPORARY
        if i == 500:
            z = sess.run(dcgan.z)
            z = np.reshape(z, [len(args.input_videos), args.vid_length, dcgan.z_dim])
            z[:, 1:, :] = z[:, :1, :]
            z = np.reshape(z, [-1, dcgan.z_dim])
            sess.run(dcgan.z.assign(z))
            current_lr *= 0.5
        # Decay learning rate, if applicable
        if freq > 0 and i % freq == freq - 1:
            current_lr *= args.lr_decay_amount
            print "SET LEARING RATE TO:", current_lr

    samples = sess.run(imgs_tensor)
    save_images(samples, [len(args.input_videos), args.vid_length],
                os.path.join(args.sample_dir, "final.png"))
    print "Saved final images"

    # Now save some stupid video-esque stuff
    samples = np.reshape(samples, full_shape)
    final_frames_folder = os.path.join(args.sample_dir, "final_frames")
    if not os.path.exists(final_frames_folder):
        # No recursive os.makedirs
        os.mkdir(final_frames_folder)
    for i in xrange(args.vid_length):
        save_images(samples[:,i,:,:,:], [1, len(args.input_videos)],
                    os.path.join(final_frames_folder, "final_frame_%03d.png" % i))

    # Now do random tweening crap
    print "Generating tween results"
    z = sess.run(dcgan.z)
    z = np.reshape(z, [len(args.input_videos), args.vid_length, dcgan.z_dim])
    tween_frames_folder = os.path.join(args.sample_dir, "tween_frames")
    if not os.path.exists(tween_frames_folder):
        # No recursive os.makedirs
        os.mkdir(tween_frames_folder)
    tween_frames = 4
    for i in xrange(args.vid_length):
        cnt = i * (tween_frames + 1)
        save_images(samples[:,i,:,:,:], [1, len(args.input_videos)],
                    os.path.join(tween_frames_folder, "tween_frame_%03d.png" % cnt))
        if i+1 < args.vid_length:
            start = z[:, i, :]
            end = z[:, i+1, :]
            for j in xrange(1, tween_frames+1):
                delta = j / float(tween_frames + 1)
                tween = end*delta + start*(1-delta)
                tween_reps = np.stack([tween] * args.vid_length, axis=1)
                tween_reps = np.reshape(tween_reps, [-1, dcgan.z_dim])
                img_tween_reps = sess.run(imgs_tensor, feed_dict={
                    dcgan.z: tween_reps,
                })
                img_tween_reps = np.reshape(img_tween_reps, full_shape)
                save_images(img_tween_reps[:,0,:,:,:], [1, len(args.input_videos)],
                            os.path.join(tween_frames_folder,
                                         "tween_frame_%03d.png" % (cnt+j)))

    # TODO FIXME
    # print "Final activation distances:"
    # ds = sess.run(activations_distance)
    # for d in ds:
    #     print d

    # print "Sanity check activation distances:"
    # final_activations = sess.run(target_activations_tensor, feed_dict={
    #     dcgan.images: samples,
    # })
    # for i in xrange(dcgan.batch_size):
    #     a = final_activations[i]
    #     print np.linalg.norm(a - target_activations[0])

if __name__ == "__main__":
    main()
