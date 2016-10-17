import urllib
import urllib2
import re
import os
import datetime
import threading
import argparse
import time

giphy_api = "http://api.giphy.com/v1/gifs/random?api_key=dc6zaTOxFJmzC"

def git_gifs(threadID, num_gifs):
    global giphy_api
    global path
    global mode

    i = 0
    failed_counter = 0
    while(i < num_gifs and failed_counter < 20):
        try:
            # query random gif/mp4, obtain url
            response = urllib2.urlopen(giphy_api)
            html = response.read()
            if (mode == "gif"):
                match = re.search("image_original_url\"\:\"([^\"]+)", html)
                ext = ".gif"
            elif (mode == "mp4"):
                match = re.search("image_mp4_url\"\:\"([^\"]+)", html)
                ext = ".mp4"
            gif_url = match.group(1).replace("\\", "")

            # prefix file name with date-time and threadID
            dt = datetime.datetime.now()
            file_name = dt.strftime("%Y-%m-%d-%H-%M-%S-") + str(threadID) + "-" + str(i)
            file_path = os.path.abspath(os.path.join(path, file_name + ext))

            # retrieve gif/mp4 from query response
            urllib.urlretrieve(gif_url, file_path)
            print "\"" + gif_url +"\" saved to " + file_path
            i += 1
            failed_counter = 0
        except:
            print "No url found from server response. Continue..."
            failed_counter += 1


class myThread(threading.Thread):
    def __init__(self, threadID, num_gifs):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.num_gifs = num_gifs
    
    def run(self):
        print "Starting thread " + str(self.threadID)
        git_gifs(self.threadID, self.num_gifs)
        print "Exiting thread " + str(self.threadID)

        
##### main ######
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Destination path for storing downloaded gifs. Default to \"./gifs_raw/\".", default = "./gifs_raw/")
parser.add_argument("--num_threads", type=int, help="Number of threads to run. Default to 50.", default = 50)
parser.add_argument("--num_items_per_thread", type=int, help="Number of times (gif or mp4) to be downloaded per thread. Default to 10.", default = 10)
parser.add_argument("--mode", type=str, help="\"gif\" or \"mp4\". Default to mp4.", default = "mp4")
args = parser.parse_args()

path = args.path
if not os.path.isdir(os.path.abspath(path)):
    os.mkdir(os.path.abspath(path))

mode = args.mode
if (mode != "gif" and mode != "mp4"):
    sys.exit("Unknown mode. Must be \"gif\" or \"mp4\".")

thread_list = []
for t in range(args.num_threads):
    thread = myThread(t, args.num_items_per_thread)
    thread.start()
    thread_list.append(thread)
    time.sleep(0.5)
    
    
    