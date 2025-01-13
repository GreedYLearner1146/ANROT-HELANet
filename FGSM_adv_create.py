from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np

######## Simple CNN that served as a basis for the adversarial samples via FGSM ############
######################### On the meta-train dataset #######################################

#initialize our optimizer and model
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

