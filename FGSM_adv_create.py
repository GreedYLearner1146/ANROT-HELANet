from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np

######## Simple CNN that served as a basis for the adversarial samples via FGSM ############
######################### On the meta-train dataset #######################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

class SimpleCNN:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape
    ##################################################
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
    ##################################################
        # first CONV => RELU => BN layer set
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
    ######################################################
        # second CONV => RELU => BN layer set
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
    #########################################################
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        return model

# import the necessary packages
from tensorflow.keras.losses import MSE
import tensorflow as tf

def generate_image_adversary(model, image, label, eps=2 / 255.0):
    # cast the image
    image = tf.cast(image, tf.float32)
    # record our gradients
    with tf.GradientTape() as tape:
        # explicitly indicate that our image should be tacked for
        # gradient updates
        tape.watch(image)
        # use our model to make predictions on the input image and
        # then compute the loss
        pred = model(image)
        loss = MSE(label, pred)  # Use MSE.
    # calculate the gradients of loss with respect to the image, then
    # compute the sign of the gradient
    gradient = tape.gradient(loss, image)
    signedGrad = tf.sign(gradient)
    # construct the image adversary
    adversary = (image + (signedGrad * eps)).numpy()  #FGSM.
    # return the image adversary to the calling function
    return adversary


# import the necessary packages
import numpy as np

def generate_adversarial_batch(model, total, images, labels, dims,
    eps=0.25):  # Change to 0.005?
    # unpack the image dimensions into convenience variables
    (h, w, c) = dims
  # we're constructing a data generator here so we need to loop
    # indefinitely
    while True:
        # initialize our perturbed images and labels
        perturbImages = []
        perturbLabels = []
        # randomly sample indexes (without replacement) from the
        # input data
        idxs = np.random.choice(range(0, len(images)), size=total,replace=False)
    # loop over the indexes
        for i in idxs:
            # grab the current image and label
            image = images[i]
            label = labels[i]
            # generate an adversarial image
            adversary = generate_image_adversary(model,
                image.reshape(1, h, w, c), label, eps=eps)
            # update our perturbed images and labels lists
            perturbImages.append(adversary.reshape(h, w, c))
            perturbLabels.append(label)
        # yield the perturbed images and labels
        yield (np.array(perturbImages), np.array(perturbLabels))

################ initialize our optimizer and model ###############################
print("[INFO] compiling model...")
opt = Adam(1e-3)
model = SimpleCNN.build(width=84, height=84, depth=3, classes=60)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
# train the simple CNN 
print("[INFO] training network...")
model.fit(np.asarray(new_X_train), np.asarray(onehot_encoded),
    #validation_data=(testX, testY),
    batch_size=64,
    epochs=100,
    verbose=1)

################################## Fine-tuning #################################################
# set (i.e., non-adversarial) again to see if performance has degraded
(loss, acc) = model.evaluate(x=np.asarray(new_X_train), y=np.asarray(onehot_encoded), verbose=0)
print("")
print("[INFO] normal evaluated images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# do a final evaluation of the model on the adversarial images
#(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
#print("[INFO] adversarial images *after* fine-tuning:")
#print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss, acc))

#################### Generate a set of adversarial from our train set ##########################
print("[INFO] generating adversarial examples with FGSM...\n")
(advX, advY) = next(generate_adversarial_batch(model, len(new_X_train),
    np.asarray(new_X_train), np.asarray(onehot_encoded), (84, 84, 3), eps=0.20))   # Can change value of epsilon here.
# re-evaluate the model on the adversarial images
(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] adversarial trained images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))

###############################################################################################

# do a final evaluation of the model on the adversarial images
(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] adversarial images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss, acc))

############################ Store trained adversarial samples in an array #####################################

train_array_adv = []

for A,B in zip(advX,onehot_encoder.inverse_transform(advY)):  # Need include inverse_transform.
  train_array_adv.append((A,B))

Adv_X_train = [x[0] for x in train_array_adv]
Adv_y_train = [x[1] for x in train_array_adv]

Adv_intytrain = []

for z in Adv_y_train:
     Adv_intytrain.append(int(z))

######################### Do the same for meta-valid set ########################

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np

#initialize our optimizer and model
print("[INFO] compiling model...")
opt = Adam(1e-3)
model = SimpleCNN.build(width=84, height=84, depth=3, classes=36)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
# train the simple CNN 
print("[INFO] training network...")
model.fit(np.asarray(new_X_val), np.asarray(onehot_encodedval),
    #validation_data=(testX, testY),
    batch_size=64,
    epochs=100,
    verbose=1)

################################ Fine-tuning ##############################################
(loss, acc) = model.evaluate(x=np.asarray(new_X_val), y=np.asarray(onehot_encodedval), verbose=0)
print("")
print("[INFO] normal evaluated images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# do a final evaluation of the model on the adversarial images
#(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
#print("[INFO] adversarial images *after* fine-tuning:")
#print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss, acc))


################### Generate a set of adversarial from our valid set ######################
print("[INFO] generating adversarial examples with FGSM...\n")
(advXval, advYval) = next(generate_adversarial_batch(model, len(new_X_val),
    np.asarray(new_X_val), np.asarray(onehot_encodedval), (84, 84, 3), eps=0.20))  # Can change value of epsilon here.
# re-evaluate the model on the adversarial images
(loss, acc) = model.evaluate(x=advXval, y=advYval, verbose=0)
print("[INFO] adversarial valid images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))

###############################################################################################

# do a final evaluation of the model on the adversarial images
(loss, acc) = model.evaluate(x=advXval, y=advYval, verbose=0)
print("[INFO] adversarial images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss, acc))

############################### Store trained adversarial samples in an array ##########################################

val_array_adv = []

for A,B in zip(advXval,onehot_encoderval.inverse_transform(advYval)):  # Need include inverse_transform.
  val_array_adv.append((A,B))

Adv_X_val = [x[0] for x in val_array_adv]
Adv_y_val= [x[1] for x in val_array_adv]

Adv_intyval = []

for z in Adv_y_val:
     Adv_intyval.append(int(z))

