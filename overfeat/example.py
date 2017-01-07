import overfeat
import numpy
from scipy.ndimage import imread
from scipy.misc import imresize
import os
import time

# read image
image = imread('test.jpg')

# resize and crop into a 231x231 image
h0 = image.shape[0]
w0 = image.shape[1]
d0 = float(min(h0, w0))
image = image[int(round((h0-d0)/2.)):int(round((h0-d0)/2.) + d0),
              int(round((w0-d0)/2.)):int(round((w0-d0)/2.) + d0), :]
image = imresize(image, (231, 231)).astype(numpy.float32)

# numpy loads image with colors as last dimension, transpose tensor
h = image.shape[0]
w = image.shape[1]
c = image.shape[2]
image = image.reshape(w * h, c)
image = image.transpose()
image = image.reshape(c, h, w)
print "Image size :", image.shape

# initialize overfeat. Note that this takes time, so do it only once if possible
overfeat_home = os.environ.get("OVERFEAT_HOME", "/OverFeat")
overfeat.init(os.path.join(overfeat_home, './data/default/net_weight_0'), 0)

# run overfeat on the image
t = time.time()
b = overfeat.fprop(image)
print(t - time.time())

# get ouput of layer n
n = 16
output = overfeat.get_output(n)

# display top 5 classes
b = b.flatten()
top = [(b[i], i) for i in xrange(len(b))]
top.sort()
print "\nTop classes :"
for i in xrange(5):
    print(overfeat.get_class_name(top[-(i+1)][1]))