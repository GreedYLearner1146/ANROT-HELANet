from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np

######## Simple CNN that served as a basis for the adversarial samples via FGSM ############
#initialize our optimizer and model
print("[INFO] compiling model...")
opt = Adam(1e-3)
model = SimpleCNN.build(width=84, height=84, depth=3, classes=36)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
# train the simple CNN on MNIST
print("[INFO] training network...")
model.fit(np.asarray(new_X_val), np.asarray(onehot_encodedval),
    #validation_data=(testX, testY),
    batch_size=64,
    epochs=100,
    verbose=1)

################################## Fine-tuning #################################################
# set (i.e., non-adversarial) again to see if performance has degraded
(loss, acc) = model.evaluate(x=np.asarray(new_X_train), y=np.asarray(onehot_encoded), verbose=0)
print("")
print("[INFO] normal testing images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# do a final evaluation of the model on the adversarial images
#(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
#print("[INFO] adversarial images *after* fine-tuning:")
#print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss, acc))

#################### generate a set of adversarial from our train set ##########################
print("[INFO] generating adversarial examples with FGSM...\n")
(advX, advY) = next(generate_adversarial_batch(model, len(new_X_train),
    np.asarray(new_X_train), np.asarray(onehot_encoded), (84, 84, 3), eps=0.20))  # Changed from 0.05, 0.1, 0.15, 0.20, 0.25, 0.30?
# re-evaluate the model on the adversarial images
(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] adversarial testing images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))

###############################################################################################

