# example of calculating the frechet inception distance in Keras.
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize

# scale an array of images to a new size
def scale_images(images, new_shape):
  images_list = list()
  for image in images:
    # resize with nearest neighbor interpolation
    new_image = resize(image, new_shape, 0)
    # store
    images_list.append(new_image)
  return asarray(images_list)

 ########### calculate frechet inception distance ##########################

def calculate_fid(model, images1, images2):
  # calculate activations
  act1 = model.predict(images1)
  act2 = model.predict(images2)
  # calculate mean and covariance statistics
  mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
  mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
  # calculate sum squared difference between means
  ssdiff = numpy.sum((mu1 - mu2)**2.0)
  # calculate sqrt of product between cov
  covmean = sqrtm(sigma1.dot(sigma2))
  # check and correct imaginary numbers from sqrt
  if iscomplexobj(covmean):
    covmean = covmean.real
  # calculate score
  fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
  return fid

##################################################################################
# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

# Shuffle and select first 10000 images of original and reconstructed images.
new_X_test = new_X_test[:10000]

X_test_reconstructed = dest[:10000]

# convert integer to floating point values
for i, j in zip(new_X_test,X_test_reconstructed):
  images1 = i.astype('float32')
  images1 = scale_images(images1, (299,299,3))
  images1 = preprocess_input(images1)
  images2 = j.astype('float32')
  images2 = scale_images(images2, (299,299,3))
  images2 = preprocess_input(images2)

#calculate fid
fid = calculate_fid(model, images1, images2)
print('FID: %.3f' % fid)

