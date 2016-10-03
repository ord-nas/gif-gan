import urllib
import urllib2
import re
import os
import datetime
import threading

giphy_api = "http://api.giphy.com/v1/gifs/random?api_key=dc6zaTOxFJmzC"
num_threads = 50
num_gifs_per_thread = 10


def git_gifs(threadID, num_gifs):
    global giphy_api
    for i in range(num_gifs):
        response = urllib2.urlopen(giphy_api)
        html = response.read()
        try:
            match = re.search("image_original_url\"\:\"([^\"]+)", html)
            gif_url = match.group(1).replace("\\", "")

            dt = datetime.datetime.now()
            ext = ".gif"
            file_name = dt.strftime("%Y-%m-%d-%H-%M-%S-") + str(threadID) + "-" + str(i)

            file_path = os.path.abspath("gifs_raw/" + file_name + ext)
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

        
# main
thread_list = []
for t in range(num_threads):
    thread = myThread(t, num_gifs_per_thread)
    thread.start()
    thread_list.append(thread)
    
print "Exiting Main Thread"