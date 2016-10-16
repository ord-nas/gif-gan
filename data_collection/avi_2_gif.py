import getopt
import os
import sys

from moviepy.editor import *


"""
Use to convert all the .avi files in a directory into gifs
If you want to change from a different format just change the
filename.endswith(".avi") part to whatever you want, moviepy can use any
extension supported by ffmpeg: .ogv, .mp4, .mpeg, .avi, .mov, etc.
"""

help_string = (
	'Usage: avi_2_gif.py -s <source_folder> -d '
	+ '<destination_folder>'
)

argv = sys.argv[1:]

try:
	opts, args = getopt.getopt(argv, "hs:d:")
except getopt.GetoptError:
	print help_string
	sys.exit(2)

source = ''
dest = ''

for opt, arg in opts:
	if opt == '-h':
		print help_string
	elif opt == '-s':
		source = arg
	elif opt == '-d':
		dest = arg

cwd = os.getcwd()
source = os.path.join(cwd, source)
dest = os.path.join(cwd, dest)
print source, dest

for filename in os.listdir(source):
	if filename.endswith(".avi"):
		abs_filename = os.path.join(source, filename)
		clip = VideoFileClip(abs_filename)
		clip.write_gif(
			os.path.join(dest, filename).replace('.avi', '.gif')
		)
