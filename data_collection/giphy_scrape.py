import urllib
import urllib2
import re
import os
import datetime
import threading
import argparse
import sys
import time

giphy_api_random = "http://api.giphy.com/v1/gifs/random?api_key=dc6zaTOxFJmzC"
giphy_api_search = "http://api.giphy.com/v1/gifs/search?q="
giphy_api_search_2 = "&api_key=dc6zaTOxFJmzC&limit=100&offset="

def get_videos(threadID, num_items):
    global giphy_api_random
    global giphy_api_search
    global path
    global num_threads
    global downloaded_videos
    global lock
    global new_counter
    global total_counter
    global keyword_counter
    global verbose

    i = 0
    failed_counter = 0
    while((i < num_items / 100) and (failed_counter < 20)):
        try:
            offset = threadID * num_items + 100 * i
            # hack to deal with the very last query
            if ((threadID == num_threads - 1) and (i == num_items / 100 - 1)):
                offset -= 1
            # query search result, obtain url
            response = urllib2.urlopen(giphy_api_search + keyword + giphy_api_search_2 + str(offset))
            html = response.read()
            url_list = re.findall("\"mp4\":\"([^\"]+giphy\.mp4)\"", html)

            for url in url_list:
                video_url = url.replace("\\", "")

                # set file name to image id
                file_name = re.search("media/([^/]+)/", video_url).group(1)
                # check if video with id is already downloaded
                # if so, skip it
                # if not, add its id to global set
                lock.acquire()
                if (file_name in downloaded_videos):
                    lock.release()
                    continue
                else:
                    downloaded_videos.add(file_name)
                    new_counter += 1
                    total_counter += 1
                    keyword_counter += 1
                    lock.release()

                file_path = os.path.abspath(os.path.join(path, file_name + ".mp4"))

                # retrieve mp4 from query response
                urllib.urlretrieve(video_url, file_path)
                if (verbose):
                    print "\"" + video_url +"\" saved to " + file_path
            i += 1
            failed_counter = 0
        except:
            print "No url found from server response. Continue..."
            failed_counter += 1


class myThread(threading.Thread):
    def __init__(self, threadID, num_items, keyword):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.num_items = num_items
        self.keyword = keyword
    
    def run(self):
        if (verbose):
            print "Starting thread " + str(self.threadID)

        get_videos(self.threadID, self.num_items)

        if (verbose):
            print "Exiting thread " + str(self.threadID)
            print "Current Active Slave Count = " + str(threading.active_count() - 2)

        
##### main ######
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Destination path for storing downloaded videos. Default to \"./data/raw/mp4s_raw/\".", default = "./data/raw/mp4s_raw/")
parser.add_argument("--num_threads", type=int, help="Number of threads to run. Default to 50.", default = 50)
parser.add_argument("--num_items_per_thread", type=int, help="Number of videos to be downloaded per thread. Default to 100.", default = 100)
parser.add_argument("--mode", type=str, help="\"random\" or \"search\". Default to mp4.", default = "random")
parser.add_argument("--verbose", type=bool, help="More output. Default to False.", default = False)
args = parser.parse_args()

path = args.path
if not os.path.isdir(os.path.abspath(path)):
    os.mkdir(os.path.abspath(path))

mode = args.mode
if (mode != "random" and mode != "search"):
    sys.exit("Unknown mode. Must be \"random\" or \"search\".")

num_items_per_thread = args.num_items_per_thread
if (num_items_per_thread % 100 != 0):
    sys.exit("Invalid num_items_per_thread. Must be an integer multiple of 100.")

verbose = args.verbose

num_threads = args.num_threads

# Global set
downloaded_videos = set()
new_counter = 0

print "Reading keywords..."
keyword_list = []
keyword_file = open("./keywords.txt")
begin_parse = False
for line in keyword_file:
    if (line[-1] == '\n'):
        k = line[:-1] # get rid of newline
    else:
        k = line
    if (k == "# keywords not tried yet"):
        begin_parse = True
        continue
    elif (begin_parse and k != ""):
        print "    " + k
        keyword_list.append(k)

print "\nScanning downloaded videos..."
# Add downloaded vidoes to global set
for folder in os.listdir("./data/raw"):
    dir_path = os.path.join("./data/raw", folder)
    print "    scanning: " + dir_path
    if os.path.isdir(dir_path):
        for file in os.listdir(dir_path):
            if file.endswith(".mp4"):
                video_id = file[:-4]
                if (video_id in downloaded_videos):
                    sys.exit("    Duplicate! id = " + video_id)
                downloaded_videos.add(video_id)

total_counter = len(downloaded_videos)

print "    Current dataset size = %d" % total_counter

print "\nStart scraping..."

lock = threading.RLock()

for keyword in keyword_list:
    print "    Scraping keyword = " + keyword
    keyword_counter = 0
    thread_list = []
    for t in range(args.num_threads):
        thread = myThread(t, num_items_per_thread, keyword)
        thread.start()
        thread_list.append(thread)
        time.sleep(0.1)
    for thread in thread_list:
        thread.join()
    print "    New videos obtained for keyword %s = %d" % (keyword, keyword_counter)

print "\nTotal new vidoes obtained = %d" % new_counter
print "Current dataset size = %d" % total_counter
print "Done"
    
    
    