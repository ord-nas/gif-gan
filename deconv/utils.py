"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division

import json
import math
import random

import cv2
import matplotlib.animation as animation
import numpy as np
import pprint
import scipy.misc
import subprocess as sp
from time import gmtime, strftime


pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w, k_t: 1/math.sqrt(k_w*k_h*k_t*x.get_shape()[-1])


def get_image(image_path, image_size, video_duration, is_crop=True,
              resize_w=64, resize_t=32, is_grayscale = False,
              should_threshold=False, should_edge=False):
    return transform(
        vidread(
            image_path, resize_w, video_duration, is_grayscale,
            should_threshold, should_edge
        ),
        image_size,
        video_duration,
        is_crop,
        resize_w,
        resize_t
    )

def save_videos(videos, num_videos, video_path, is_grayscale=False):
    videos = inverse_transform(videos)

    h = videos[0].shape[0]
    t = videos[0].shape[2]
    vid_dim = int(math.ceil(num_videos**0.5) * h)
    out = cv2.VideoWriter(
        '{}_samples.mp4'.format(video_path),
        0x20,
        25.0,
        (vid_dim, vid_dim)
    )

    # Merge all the videos into one
    video = np.random.rand(vid_dim, vid_dim, t, 3)
    for i, sample in enumerate(videos[0:num_videos]):
        x_pos = i % (vid_dim // h)
        y_pos = i // (vid_dim // h)
        if is_grayscale:
            # Repeat same frame 3 times across color dimension to turn c_dim=1
            # into a color video that's still B+W
            sample = np.repeat(sample, 3, axis=3)
        video[y_pos*h:(y_pos+1)*h, x_pos*h:(x_pos+1)*h, :, :] = sample

    for i in range(t):
        out.write(np.uint8(255 * video[:,:,i,:]))
    out.release()

def vidread(path, resize_w, num_frames, is_grayscale=False,
            should_threshold=False, should_edge=False):
    if is_grayscale:
        video = np.empty([num_frames, resize_w, resize_w, 1])
    else:
        video = np.empty([num_frames, resize_w, resize_w, 3])

    cap = cv2.VideoCapture(path)

    i = 0
    while(cap.isOpened() and i < num_frames):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.resize(
                frame, (resize_w, resize_w), interpolation=cv2.INTER_CUBIC
            )
            if should_threshold:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
                frame = np.expand_dims(frame, axis=2)
            elif should_edge:
                frame = cv2.Canny(frame, 100, 200)
                frame = np.expand_dims(frame, axis=2)

            video[i] = frame
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Source video doesn't have enough frames, None is handled further up
    if i < num_frames:
        return None

    cap.release()
    cv2.destroyAllWindows()

    capture = cv2.VideoCapture(path)
    
    # Want [t, w, h, c] -> [w, h, t, c]
    video = np.swapaxes(video, 0, 2) # [t, w, h, c] -> [h, w, t, c]
    video = np.swapaxes(video, 0, 1) # [h, w, t, c] -> [w, h, t, c]

    return video


# def merge(videos, size):
#     h, w = images.shape[1], images.shape[2]
#     img = np.zeros((h * size[0], w * size[1], 3))
#     for idx, image in enumerate(images):
#         i = idx % size[1]
#         j = idx // size[1]
#         img[j*h:j*h+h, i*w:i*w+w, :] = image

#     return img


# def vidsave(videos, size, path):
#     return scipy.misc.imsave(path, merge(images, size))


def center_crop(x, crop_h, crop_t, crop_w=None, resize_w=64, resize_t=32):
    if crop_w is None:
        crop_w = crop_h
    h, w, t = x.shape[:2]
    i = int(round((h - crop_h)/2.))
    j = int(round((w - crop_w)/2.))
    k = int(round((t - crop_t)/2.))
    return scipy.misc.imresize(
        x[i:i+crop_h, j:j+crop_w, k:k+crop_t], [resize_w, resize_w, resize_t]
    )


def transform(video, npx=64, nf=32, is_crop=True, resize_w=64, resize_t=32):
    # npx : # of pixels width/height of image
    # nf: # of frames of video
    if video is None:
        return None
    if is_crop:
        cropped_video = center_crop(
            video, npx, nf, resize_w=resize_w, resize_t=resize_t
        )
    else:
        cropped_video = video
    return np.array(cropped_video)/127.5 - 1.


def inverse_transform(videos):
    return (videos+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {
                "sy": 1,
                "sx": 1,
                "depth": depth,
                "w": ['%.2f' % elem for elem in list(B)]
            }
            gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
            beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({
                        "sy": 1,
                        "sx": 1,
                        "depth": W.shape[0],
                        "w": ['%.2f' % elem for elem in list(w)]
                    })

                lines += """
                    var layer_%s = {
                        "layer_type": "fc", 
                        "sy": 1, "sx": 1, 
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (
                        layer_idx.split('_')[0],
                        W.shape[1],
                        W.shape[0],
                        biases,
                        gamma,
                        beta,
                        fs
                    )
            else:
                fs = []
                for w_ in W:
                    fs.append({
                        "sy": 5,
                        "sx": 5,
                        "depth": W.shape[3],
                        "w": ['%.2f' % elem for elem in list(w_.flatten())]
                    })

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv", 
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (
                        layer_idx,
                        2**(int(layer_idx)+2),
                        2**(int(layer_idx)+2),
                        W.shape[0],
                        W.shape[3],
                        biases,
                        gamma,
                        beta,
                        fs
                    )
        layer_f.write(" ".join(lines.replace("'","").split()))


def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)


def visualize(sess, dcgan, config, option):
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
