import re
import sys

times = []

while True:
    inpt = sys.stdin.readline()
    if inpt == '':
        break
    m = re.search("time: ([0-9.]*),", inpt)
    if m is not None:
        times.append(float(m.group(1)))

deltas = [b-a for (a, b) in zip(times, times[1:]) if b-a < 100]

avg = sum(deltas) / len(deltas)
print "Average:", avg
print "Epoch hrs:", avg * 3165 / 60 / 60
