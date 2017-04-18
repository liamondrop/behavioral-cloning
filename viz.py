import os
import csv
import time
import random
import numpy as np
from scipy.misc import imread, imsave
from keras.models import load_model
from keras import backend as K

model = load_model('./model.h5')
input_img = model.input
img_height, img_width = input_img.shape[1:3]

layer_dict = {layer.name: layer for layer in model.layers}
layer_name = 'conv2d_5'
layer_output = layer_dict[layer_name].output
n_filters = int(layer_output.shape[-1])

DATA_DIR = os.path.abspath('data')
LOG_FILE = 'driving_log.csv'

# Load the image_data from the driving log
image_data = []
with open(os.path.join(DATA_DIR, LOG_FILE)) as handle:
    reader = csv.reader(handle)
    for row in reader:
        image_data.append(row)

# Get a random line from the driving log csv
line = image_data[random.randint(0, len(image_data))]
image_file = os.path.join(DATA_DIR, line[0])
# image_file = "/Users/bowerslm/Dev/sdc/behavioral-cloning/data/IMG/center_2017_04_17_00_39_56_559.jpg"

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

kept_filters = []
for filter_index in range(0, n_filters):
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, img_height, img_width))
    else:
        input_img_data = np.random.random((1, img_height, img_width, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # input_img_data = imread(image_file).astype(np.float32)
    # input_img_data = input_img_data.reshape((-1,) + input_img_data.shape)

    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

n = 3

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top nxn filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

margin = 5
width = n * int(img_width) + (n - 1) * margin
height = n * int(img_height) + (n - 1) * margin
stitched_filters = np.zeros((height, width, 3))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_height + margin) * i: (img_height + margin) * i + img_height,
                         (img_width + margin) * j: (img_width + margin) * j + img_width, :] = img

# save the result to disk
imsave('stitched_filters_%s_%dx%d.png' % (layer_name, n, n), stitched_filters)
print('IMAGE FILE', image_file)
