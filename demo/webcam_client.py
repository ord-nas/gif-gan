import cv2
import argparse
import os
import math
import numpy as np
import subprocess
import sys
import glob

# Params for algorithm
parser = argparse.ArgumentParser()
# Input/output params
parser.add_argument("--capture_device", type=int, default=0, help="Device id to use as webcam")
parser.add_argument("--local_output_directory", required=True, help="Local directory to place output")
# Remote params
parser.add_argument("--remote_username", required=True, help="Username to use for remote login")
parser.add_argument("--remote_host", default="ug55.eecg.utoronto.ca", help="Remote hosts to access")
parser.add_argument("--remote_target_directory", default="/thesis0/yccggrp/demo/webcam", help="Remote directory to place output")
parser.add_argument("--remote_src_directory", default="/thesis0/yccggrp/youngsan/gif-gan/demo", help="Remote directory to find source")
# Params for the Haar Cascade Classifier
parser.add_argument("--opencv_data_dir", default="classifier_configs", help="Directory from which to load classifier config file")
parser.add_argument("--classifier_config_file", default="haarcascade_frontalface_alt2.xml", help="Classifier config file")
parser.add_argument("--classifier_scale_factor", type=float, default=1.1, help="cc.detectMultiScale(scaleFactor)")
parser.add_argument("--classifier_min_neighbors", type=int, default=4, help="cc.detectMultiScale(minNeighbors)")
parser.add_argument("--classifier_min_size", type=int, default=32, help="cc.detectMultiScale(minSize)")
parser.add_argument("--classifier_max_size_factor", type=float, default=1.0, help="Multiplier on image side length to determine cc.detectMultiScale(maxSize)")
parser.add_argument("--classifier_flags", type=int, default=cv2.CASCADE_DO_CANNY_PRUNING, help="cc.detectMultiScale(flags)")
# Params for determining how to size the bounding box around the face
parser.add_argument("--target_width", type=int, default=64, help="Width of the cropped face video")
parser.add_argument("--target_height", type=int, default=64, help="Height of the cropped face video")
parser.add_argument("--bounding_box_scaling_factor", type=float, default=1.0, help="Amount to scale the haar cascade bounding box before cropping")
parser.add_argument("--resize_interpolation_method", type=eval, default=cv2.INTER_LINEAR, help="cv2.resize(interpolation)")

backups = ["/home/sandro/Desktop/keepers/sandro_2", "/home/sandro/Desktop/keepers/anastasia"]

def get_face_from_webcam(webcam, cc, args):
    cv2.namedWindow("Webcam")
    cv2.moveWindow("Webcam", 400, 0)
    while True:
        assert webcam.isOpened()
        ret, original_im = webcam.read()
        assert ret
        im = np.copy(original_im)

        # Get face detections on this frame
        side = math.sqrt(im.shape[0] * im.shape[1])
        minlen = args.classifier_min_size
        maxlen = int(side * args.classifier_max_size_factor)
        features = list(cc.detectMultiScale(
            im, args.classifier_scale_factor, args.classifier_min_neighbors,
            args.classifier_flags, (minlen, minlen), (maxlen, maxlen)))
        valid_face = False
        if features:
            best = max(features, key=lambda d: d[2] * d[3])
            # Expand the box along one axis so the aspect ratio is correct
            required_aspect_ratio = float(args.target_width)/float(args.target_height)
            actual_aspect_ratio = float(best[2])/float(best[3])
            scaling = required_aspect_ratio / actual_aspect_ratio
            x_scaling = scaling if scaling > 1.0 else 1.0
            y_scaling = 1.0/scaling if scaling <= 1.0 else 1.0
            # Find the centre of the box, and expand outward from there
            centre_x = best[0] + best[2]/2.0
            centre_y = best[1] + best[3]/2.0
            assert(centre_x >= 0 and centre_x < im.shape[1])
            assert(centre_y >= 0 and centre_y < im.shape[0])
            # We expand to get the right aspect ratio, and on top of that, we
            # add an additional command-line-specified scaling factor
            f = args.bounding_box_scaling_factor
            x1 = int(round(x_scaling * f * (best[0] - centre_x) + centre_x))
            y1 = int(round(y_scaling * f * (best[1] - centre_y) + centre_y))
            x2 = int(round(x_scaling * f * (best[0] + best[2] - centre_x) + centre_x))
            y2 = int(round(y_scaling * f * (best[1] + best[3] - centre_y) + centre_y))
            # Check if the box is still within the image
            if x1 >= 0 and y1 >= 0 and x2 < im.shape[1] and y2 < im.shape[0]:
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.rectangle(im, (x1, y1), (x2, y2), (255, 255, 255), 2)
                valid_face = True

        cv2.imshow("Webcam", im[:, ::-1, :])
        key = cv2.waitKey(50)
        if key == 10 and valid_face:
            return (original_im[y1:y2+1,x1:x2+1], key)
        elif key == ord('q'):
            return (None, None)
        elif key in [ord(str(i)) for i in xrange(len(backups))]:
            return (None, int(chr(key)))

def get_face_from_image(path, cc, args):
    original_im = cv2.imread(path)
    im = np.copy(original_im)

    # Get face detections on this frame
    side = math.sqrt(im.shape[0] * im.shape[1])
    minlen = args.classifier_min_size
    maxlen = int(side * args.classifier_max_size_factor)
    features = list(cc.detectMultiScale(
        im, args.classifier_scale_factor, args.classifier_min_neighbors,
        args.classifier_flags, (minlen, minlen), (maxlen, maxlen)))
    assert features
    best = max(features, key=lambda d: d[2] * d[3])
    # Expand the box along one axis so the aspect ratio is correct
    required_aspect_ratio = float(args.target_width)/float(args.target_height)
    actual_aspect_ratio = float(best[2])/float(best[3])
    scaling = required_aspect_ratio / actual_aspect_ratio
    x_scaling = scaling if scaling > 1.0 else 1.0
    y_scaling = 1.0/scaling if scaling <= 1.0 else 1.0
    # Find the centre of the box, and expand outward from there
    centre_x = best[0] + best[2]/2.0
    centre_y = best[1] + best[3]/2.0
    assert(centre_x >= 0 and centre_x < im.shape[1])
    assert(centre_y >= 0 and centre_y < im.shape[0])
    # We expand to get the right aspect ratio, and on top of that, we
    # add an additional command-line-specified scaling factor
    f = args.bounding_box_scaling_factor
    x1 = int(round(x_scaling * f * (best[0] - centre_x) + centre_x))
    y1 = int(round(y_scaling * f * (best[1] - centre_y) + centre_y))
    x2 = int(round(x_scaling * f * (best[0] + best[2] - centre_x) + centre_x))
    y2 = int(round(y_scaling * f * (best[1] + best[3] - centre_y) + centre_y))
    # Check if the box is still within the image
    assert x1 >= 0 and y1 >= 0 and x2 < im.shape[1] and y2 < im.shape[0]
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv2.rectangle(im, (x1, y1), (x2, y2), (255, 255, 255), 2)

    cv2.imshow("Webcam", im)
    cv2.moveWindow("Webcam", 400, 0)
    key = cv2.waitKey(50)
    return original_im[y1:y2+1,x1:x2+1]

def show_reconstruction_image(filename, interp_method):
    reconstruction = cv2.imread(filename)
    reconstruction = cv2.resize(reconstruction,(800,800), interpolation=interp_method)
    cv2.namedWindow("Reconstruction")
    cv2.imshow("Reconstruction", reconstruction)
    cv2.moveWindow("Reconstruction", 400, 0)
    key = -1
    while key == -1:
        key = cv2.waitKey(50)
    cv2.destroyWindow("Reconstruction")
    return key

def show_progress_video(filename, interp_method):
    cv2.namedWindow("Progress")
    cv2.moveWindow("Progress", 400, 0)
    while True:
        cap = cv2.VideoCapture(filename)
        assert cap.isOpened()
        ret, im = cap.read()
        assert ret
        im = cv2.resize(im, (800,800), interpolation=interp_method)
        for _ in xrange(20):
            cv2.imshow("Progress", im)
            key = cv2.waitKey(100)
            if key != -1:
                cv2.destroyWindow("Progress")
                return key
        while cap.isOpened():
            ret, next_im = cap.read()
            if not ret:
                break
            im = next_im
            im = cv2.resize(im, (800,800), interpolation=interp_method)
            cv2.imshow("Progress", im)
            key = cv2.waitKey(100)
            if key != -1:
                cv2.destroyWindow("Progress")
                return key
        for _ in xrange(20):
            cv2.imshow("Progress", im)
            key = cv2.waitKey(100)
            if key != -1:
                cv2.destroyWindow("Progress")
                return key

def show_path(filename, interp_method):
    name, _ = os.path.splitext(os.path.basename(filename))
    assert name[:5] == "path_"
    name = name[5:]
    name = name.replace("_", " ")
    name = name[0].upper() + name[1:]
    cv2.namedWindow(name)
    cv2.moveWindow(name, 400, 0)
    while True:
        cap = cv2.VideoCapture(filename)
        while cap.isOpened():
            ret, im = cap.read()
            if not ret:
                break
            im = cv2.resize(im, (800,800), interpolation=interp_method)
            cv2.imshow(name, im)
            key = cv2.waitKey(100)
            if key != -1:
                cv2.destroyWindow(name)
                return key    
            
def carousel(fns):
    assert fns
    i = 0
    while True:
        key = fns[i]()
        if key == ord('n') and i < len(fns) - 1:
            i += 1
        elif key == ord('p') and i > 0:
            i -= 1
        elif key == 10:
            return
            
def main():
    args = parser.parse_args()
    config = os.path.abspath(os.path.join(args.opencv_data_dir, args.classifier_config_file))
    cc = cv2.CascadeClassifier(config)
    #filename = "/home/sandro/Documents/ECE496/gif-gan/data_collection/3o7TKtZqP4MyMG5QC4.mp4"
    webcam = cv2.VideoCapture(args.capture_device)

    # Make output directory if it doesn't exist
    if not os.path.exists(args.local_output_directory):
        os.makedirs(args.local_output_directory)
    
    while True:
        (face, index) = get_face_from_webcam(webcam, cc, args)
        if face is None and index is None:
            return

        if face is not None:
            cv2.imshow("Webcam", face)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            cv2.waitKey(50)

            # Do scp
            capture_file = os.path.join(args.local_output_directory, "webcam_face_capture.png")
            face = cv2.resize(face,(args.target_width,args.target_height), interpolation=args.resize_interpolation_method)
            cv2.imwrite(capture_file, face)
            host = "%s@%s" % (args.remote_username, args.remote_host)
            remote_path = "%s:%s" % (host, args.remote_target_directory)
            ret = subprocess.call(["scp", capture_file, remote_path])
            assert ret == 0

            # Execute remote command
            ret = subprocess.call(["ssh", host, "%s/run_webcam_demo" % args.remote_src_directory])
            assert ret == 0

            # Retrieve result
            src_path = "%s/output" % remote_path
            ret = subprocess.call(["rm", "-rf", "%s/output" % args.local_output_directory])
            assert ret == 0
            ret = subprocess.call(["scp", "-r", src_path, args.local_output_directory])
            assert ret == 0
        else:
            assert index is not None
            subprocess.call(["cp", "-r", os.path.join(backups[index], "output"), args.local_output_directory])
            face = cv2.imread(os.path.join(backups[index], "webcam_face_capture.png"))

        # Show result
        face = cv2.resize(face,(64,64), interpolation=args.resize_interpolation_method)
        face = cv2.resize(face,(200,200), interpolation=args.resize_interpolation_method)
        cv2.namedWindow("Original")
        cv2.imshow("Original", face)
        cv2.moveWindow("Original", 0, 0)

        # Show reconstruction
        f1 = lambda: show_reconstruction_image("%s/output/final.png" % args.local_output_directory,
                                               args.resize_interpolation_method)

        # Show the progress vid
        f2 = lambda: show_progress_video("%s/output/progress.mp4" % args.local_output_directory,
                                         args.resize_interpolation_method)

        # Show the path videos
        paths = sorted(glob.glob("%s/output/path_*.mp4" % args.local_output_directory))
        path_fs = [lambda p=p: show_path(p, args.resize_interpolation_method) for p in paths]

        carousel([f1, f2] + path_fs)
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()


