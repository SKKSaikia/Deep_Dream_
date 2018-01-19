'''
#deep_dream
@SKKSaikia

Majority of code ported from @giuseppebonaccorso
Try implementing in different models.
'''

#import libraries
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1000)

import multiprocessing
import warnings

import keras.backend as K
import tensorflow as tf

from keras.applications.vgg19 import VGG19
#from keras.applications.resnet50 import ResNet50       for importing resnet
from keras.applications.imagenet_utils import preprocess_input

from scipy.optimize import minimize

from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.transform import pyramid_gaussian, rescale

#Enable GPU support
use_gpu = True

config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(), inter_op_parallelism_threads=multiprocessing.cpu_count(), allow_soft_placement=True, device_count = {'CPU' : 1, 'GPU' : 1 if use_gpu else 0})

session = tf.Session(config=config)
K.set_session(session)


#Image
def show_image(image):
    fig, ax = plt.subplots(figsize=(18, 15))
    ax.imshow(image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def preprocess_image(image):
    return preprocess_input(np.expand_dims(image.astype(K.floatx()), 0))

def postprocess_image(image):
    image[:, :, :, 0] += 103.939
    image[:, :, :, 1] += 116.779
    image[:, :, :, 2] += 123.68
    return np.clip(image[:, :, :, ::-1], 0, 255).astype('uint8')[0]

#the example call of duty image being loaded
original_image = 'cod.jpg'
final_image = 'cod_dream.jpg'

original_image_array = imread(original_image)
show_image(original_image_array)

# Ignore some warnings from scikit-image
warnings.simplefilter("ignore")

# Create gaussian pyramid
original_image_as_float = img_as_float(original_image_array)
pyramid = list(pyramid_gaussian(original_image_as_float, downscale=2, max_layer=5))

# Convert each image to unsigned byte (0-255)
for i, image in enumerate(pyramid):
    pyramid[i] = img_as_ubyte(pyramid[i])
    print('Image {}) Size: {}'.format(i, pyramid[i].shape))

convnet = VGG19(include_top=False, weights='imagenet')

layers = {
    'block5_conv1': 0.001,
    'block5_conv2': 0.001,
}

image_l2_weight = 0.005

loss_tensor = 0.0

for layer, weight in layers.items():
    loss_tensor += (-weight * K.sum(K.square(convnet.get_layer(layer).output)))

loss_tensor += image_l2_weight * K.sum(K.square(convnet.layers[0].input))

_loss_function = K.function(inputs=[convnet.layers[0].input], outputs=[loss_tensor])

loss_gradient = K.gradients(loss=loss_tensor, variables=[convnet.layers[0].input])
_gradient_function = K.function(inputs=[convnet.layers[0].input], outputs=loss_gradient)

def loss(x, shape):
    return _loss_function([x.reshape(shape)])[0]

def gradient(x, shape):
    return _gradient_function([x.reshape(shape)])[0].flatten().astype(np.float64)

def process_image(image, iterations=2):
    # Create bounds
    bounds = np.ndarray(shape=(image.flatten().shape[0], 2))
    bounds[:, 0] = -128.0
    bounds[:, 1] = 128.0

    # Initial value
    x0 = image.flatten()

    # Perform optimization
    result = minimize(fun=loss, x0=x0,args=list(image.shape), jac=gradient, method='L-BFGS-B', bounds=bounds, options={'maxiter': iterations})
    return postprocess_image(np.copy(result.x.reshape(image.shape)))

processed_image = None

for i, image in enumerate(pyramid[::-1]):
    print('Processing pyramid image: {} {}'.format(len(pyramid)-i, image.shape))

    if processed_image is None:
        processed_image = process_image(preprocess_image(image))
    else:
        h, w = image.shape[0:2]
        ph, pw = processed_image.shape[0:2]
        rescaled_image = rescale(processed_image, order=5, scale=(float(h)/float(ph), float(w)/float(pw)))
        combined_image = img_as_ubyte((1.2*img_as_float(image) + 0.8*rescaled_image) / 2.0)
        processed_image = process_image(preprocess_image(combined_image), iterations=5)

show_image(processed_image)

imsave(final_image, processed_image)

nb_frames = 3000

h, w = processed_image.shape[0:2]

for i in range(nb_frames):
    rescaled_image = rescale(processed_image, order=5, scale=(1.1, 1.1))
    rh, rw = rescaled_image.shape[0:2]

    # Compute the cropping limits
    dh = int((rh - h) / 2)
    dw = int((rw - w) / 2)

    dh1 = dh if dh % 2 == 0 else dh+1
    dw1 = dw if dw % 2 == 0 else dw+1

    # Compute an horizontal pan
    pan = int(45.0*np.sin(float(i)*np.pi/60))

    zoomed_image = rescaled_image[dh1:rh-dh, dw1+pan:rw-dw+pan, :]
    processed_image = process_image(preprocess_image(img_as_ubyte(zoomed_image)), iterations=2)

    imsave(final_image + 'img_' + str(i+1) + '.jpg', processed_image)
