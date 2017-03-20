import getopt
import os
import random
import sys

argv = sys.argv[1:]

help_string = ''

try:
    opts, args = getopt.getopt(argv, "a:b:d:n:", ["url_a=", "url_b="])
except getopt.GetoptError:
    print help_string
    sys.exit(2)

source = ''
dest = ''

print opts

for opt, arg in opts:
    if opt == '-a':
        source1 = arg
    elif opt == '-b':
        source2 = arg
    elif opt == '-d':
        dest = arg
    elif opt == '-n':
        num = int(arg)
    elif opt == '--url_a':
        url_a = arg
    elif opt == '--url_b':
        url_b = arg

cwd = os.getcwd()
source1 = os.path.join(cwd, source1)
source2 = os.path.join(cwd, source2)
dest = os.path.join(cwd, dest)

with open(source1, 'r') as fa, open(source2, 'r') as fb:
    lines_a = fa.readlines()
    print len(lines_a)
    lines_b = fb.readlines()
    print len(lines_b)
    lines_out = ['image_A_url,image_B_url,swap']
    for i in range(num):
        swap = random.randint(0,1)
        a = "https://yccggrp.firebaseapp.com/mturk/" + url_a + lines_a[i].replace('\n','')
        b = "https://yccggrp.firebaseapp.com/mturk/" + url_b + lines_b[i].replace('\n','')
        if swap == 0:
            lines_out.append(','.join([a,b,str(swap)]))
        else:
            lines_out.append(','.join([b,a,str(swap)]))
            
with open(dest, 'w') as f:
    for i in range(len(lines_out)):
        f.write(lines_out[i]+'\n')
