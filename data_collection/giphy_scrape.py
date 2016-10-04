import urllib
import urllib2
import re
import os
import datetime
import threading
import argparse

giphy_api = "http://api.giphy.com/v1/gifs/random?api_key=dc6zaTOxFJmzC"

def git_gifs(threadID, num_gifs):
    global giphy_api
    global path
    for i in range(num_gifs):
        try:
            response = urllib2.urlopen(giphy_api)
            html = response.read()
            match = re.search("image_original_url\"\:\"([^\"]+)", html)
            gif_url = match.group(1).replace("\\", "")

            dt = datetime.datetime.now()
            ext = ".gif"
            file_name = dt.strftime("%Y-%m-%d-%H-%M-%S-") + str(threadID) + "-" + str(i)

            file_path = os.path.abspath(os.path.join(path, file_name + ext))
            urllib.urlretrieve(gif_url, file_path)
            print "\"" + gif_url +"\" saved to " + file_path
        except:
            print "No gif url found from server response. Continue..."


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
parser.add_argument("--path", type=str, help="Destination path for storing downloaded GIFs. Default to \"./gifs_raw/\".", default = "./gifs_raw/")
parser.add_argument("--num_threads", type=int, help="Number of threads to run. Default to 50.", default = 50)
parser.add_argument("--num_gifs_per_thread", type=int, help="Number of GIFs to be downloaded per thread. Default to 10.", default = 10)
args = parser.parse_args()

path = args.path
if not os.path.isdir(os.path.abspath(path)):
    os.mkdir(os.path.abspath(path))

thread_list = []
for t in range(args.num_threads):
    thread = myThread(t, args.num_gifs_per_thread)
    thread.start()
    thread_list.append(thread)
    
    
    