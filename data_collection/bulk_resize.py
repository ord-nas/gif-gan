import getopt
import os
import sys

import numpy as np
import cv2


argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "s:d:w:l:")
except getopt.GetoptError:
    print '-s source_folder -d destination_folder -w new_width -l new_duration'
    sys.exit(2)

source = ''

for opt, arg in opts:
    if opt == '-s':
        source = arg
    elif opt == '-d':
        dest = arg
    elif opt == '-w':
        width = int(arg)
    elif opt == '-l':
        length = int(arg)

cwd = os.getcwd()
source = os.path.join(cwd, source)
dest = os.path.join(cwd, dest)

filenames = ['%s/%s' % (source, filename) for filename in os.listdir(source)]

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'H264')

for filename in filenames:

    cap = cv2.VideoCapture(filename)
    new_filename = os.path.join(dest, os.path.basename(filename))
    out = cv2.VideoWriter(new_filename, fourcc, 25, (width, width))
    frames = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.resize(
                frame, (width, width), interpolation=cv2.INTER_CUBIC
            )
            frames.append(frame)

            # cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    if len(frames) < length:
        continue
    else:
        frames = frames[0:length]
    for frame in frames:
        out.write(frame)

    cap.release()
    out.release()

cv2.destroyAllWindows()