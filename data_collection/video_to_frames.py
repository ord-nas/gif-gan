import os
import sys
import argparse
import re
import subprocess
import cv2

# Author: Charles Chen (charles.chen@mail.utoronto.ca)
# This script requires ffmpeg installed to run!

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--num_videos", type=int, default = 10)
parser.add_argument("--input_type", type=str, default = "gif")
parser.add_argument("--output_type", type=str, default = "png")
parser.add_argument("--frame_rate", type=int, default = 10)
parser.add_argument("--crop_faces", type=bool, default = False)
parser.add_argument("--facedetect_script_path", type=str, default = "./facedetect.py")
args = parser.parse_args()

# set parameters
input_path = os.path.abspath(args.input_path)
if not os.path.isdir(input_path):
    sys.exit("Invalid input path. Input path must be a directory.")

output_path = os.path.abspath(args.output_path)
if not os.path.isdir(output_path):
    os.mkdir(output_path)

num_videos = args.num_videos

input_type = args.input_type
if (input_type != "mp4" and input_type != "gif"):
    sys.exit("Unknown input type \"" + input_type + "\".")

output_type = args.output_type
if (output_type != "png" and output_type != "gif" and output_type != "jpg"):
    sys.exit("Unknown output type \"" + output_type + "\".")

frame_rate = args.frame_rate
if (frame_rate <= 0):
    sys.exit("Invalid frame rate.")

crop_faces = args.crop_faces
if crop_faces:
    facedetect_script_path = os.path.abspath(args.facedetect_script_path)
    if not os.path.isfile(facedetect_script_path):
        sys.exit("facedetec script doesn't exist.")

# main loop
count = 1
for file in os.listdir(input_path):
    input_file_path = os.path.join(input_path, file)
    if os.path.isfile(input_file_path) and file.endswith("." + input_type):
        file_name = re.search("(.+)\." + input_type, file).group(1)
        output_file_path = os.path.join(output_path, file_name)
        if not os.path.isdir(output_file_path):
            os.mkdir(output_file_path)
            cmd = " ".join(["ffmpeg -i", input_file_path, "-r", str(frame_rate), os.path.join(output_file_path, "frame_%d." + output_type)])
            print "\n" + cmd
            os.system(cmd)

            if (crop_faces):
                for image in os.listdir(output_file_path):
                    image_path = os.path.join(output_file_path, image)
                    args = ["python", facedetect_script_path, "--best", image_path]
                    print " ".join(args)
                    p = subprocess.Popen(args, stdout=subprocess.PIPE)
                    stdout = p.communicate()[0]
                    if (stdout != ""): # a face has been detected, crop the image
                        [x, y, w, h] = [int(i) for i in stdout.split(" ")]
                        img = cv2.imread(image_path)
                        cropped_img =  img[y:y+h, x:x+w]
                        cv2.imwrite(image_path, cropped_img)
                    else: # no face detected, delete image
                        os.remove(image_path)

                if os.listdir(output_file_path) == []:
                    os.rmdir(output_file_path)

                    

        
        count += 1

    if count > num_videos:
        break
