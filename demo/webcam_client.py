import cv2
import argparse
import os
import math
import numpy as np
import subprocess
import sys

# Params for algorithm
parser = argparse.ArgumentParser()
# Input/output params
parser.add_argument("--capture_device", type=int, default=0, help="Device id to use as webcam")
parser.add_argument("--local_output_directory", required=True, help="Local directory to place output")
# Remote params
parser.add_argument("--remote_username", required=True, help="Username to use for remote login")
parser.add_argument("--remote_host", default=["ug55.eecg.utoronto.ca"], nargs='+', help="Remote hosts to access")
parser.add_argument("--remote_target_directory", default="/thesis0/yccggrp/demo/webcam", help="Remote directory to place output")
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

def get_face_from_webcam(webcam, cc, args):
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

        cv2.imshow("Webcam", im)
        key = cv2.waitKey(50)
        if key == 10 and valid_face:
            return original_im[y1:y2+1,x1:x2+1]

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
        face = get_face_from_webcam(webcam, cc, args)
        cv2.imshow("Webcam", face)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.waitKey(50)

        # Do scp
        capture_file = os.path.join(args.local_output_directory, "webcam_face_capture.png")
        #capture_file = "/home/sandro/Documents/ECE496/my-face/turning/target.png"
        face = cv2.resize(face,(args.target_width,args.target_height), interpolation=args.resize_interpolation_method)
        cv2.imwrite(capture_file, face)
        host = "%s@%s" % (args.remote_username, args.remote_host[0])
        remote_path = "%s:%s" % (host, args.remote_target_directory)
        ret = subprocess.call(["scp", capture_file, remote_path], stdout=sys.stdout, stderr=sys.stderr)
        assert ret == 0

        # Execute remote command
        ret = subprocess.call(["ssh", host, "/thesis0/yccggrp/demo/webcam/run_webcam_demo"],
                              stdout=sys.stdout, stderr=sys.stderr)
        assert ret == 0

        # Retrieve result
        src_path = "%s/output" % remote_path
        ret = subprocess.call(["scp", "-r", src_path, args.local_output_directory])
        assert ret == 0
        reconstruction = cv2.imread("%s/output/final.png" % args.local_output_directory)

        # Show result
        face = cv2.resize(face,(200,200), interpolation=args.resize_interpolation_method)
        #face = np.zeros((400,400,3), dtype=np.uint8)
        reconstruction = cv2.resize(reconstruction,(200,200), interpolation=args.resize_interpolation_method)
        cv2.namedWindow("Original")
        cv2.namedWindow("Reconstruction")
        cv2.imshow("Original", face)
        cv2.imshow("Reconstruction", reconstruction)
        cv2.moveWindow("Reconstruction", 200, 0)
        key = -1
        while key == -1:
            key = cv2.waitKey(50)
        cv2.destroyAllWindows()
        return
        
if __name__ == "__main__":
    main()


