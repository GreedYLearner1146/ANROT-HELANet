###################### Sampling call from class ####################################

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon  #Reparameterization Trick

#################################### The encoder ########################################

latent_dim = 1024 #2

encoder_inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)  # Mean
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)  # Std
z = Sampling()([z_mean, z_log_var])  # Sampling
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

#################################### The decoder ########################################

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8 * 8 * 32, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 32))(x) # 7 7 64 64 32
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")


########### Train the VAE with various distance (KL divergence, Hellinger distance, Wasserstein Distance ################

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    tf.keras.losses.MSE(data, reconstruction),  # Replace with mse loss in cifar-10. (tf.keras.losses.MSE). #binary_crossentropy for mnist.
                )
            )
             # Modify the first three lines here in our HELA_VFA framework.

             ############### KL divergence ##############################
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=-1))
            total_loss = reconstruction_loss + kl_loss

             ############### Hellinger distance ##############################
            #N1 = (0.125)*z_log_var
            #D1 = (0.5)*ops.log(0.5) + (0.25)*z_log_var + 5e-4
            #N2 = -0.25*ops.square(z_mean)
            #D2 = (ops.exp(z_log_var) + 1)**(0.5) + 5e-4
            #A = 0.5*(ops.divide(N1,D1))*ops.exp(ops.divide(N2,D2))
            #kl_loss = ops.abs(ops.mean(ops.sum(A, axis = -1)))
            #total_loss = reconstruction_loss + ops.log(kl_loss)

            ############### Wasserstein Distance ##############################
            #kl_loss = ops.square(z_mean) + ops.exp(z_log_var) + 1 - 2*ops.exp(0.5*z_log_var)
            #kl_loss = ops.mean(ops.sum(kl_loss, axis=-1))
            #total_loss = reconstruction_loss + kl_loss


        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

############################## Run the training (miniImageNet) ###########################
vae.fit(np.asarray(new_X_train), epochs=500, batch_size=128)  

############################## vae inference (miniImageNet) ###########################
x_test_bn,_,_ = vae.encoder.predict(np.asarray(new_X_test))
x_test_reconstructed = vae.decoder.predict(x_test_bn)

# Store results in array format

dest = []

for dst in x_test_reconstructed:
    dest.append(dst)

################# code for displaying multiple images in one figure ######################

#import libraries
import cv2
from matplotlib import pyplot as plt

# create figure
fig = plt.figure(figsize=(7, 7))

# setting values to rows and column variables
rows = 6
columns = 6

# reading images
Image1 = dest[32]
Image2 = dest[87]
Image3 = dest[67]
Image4 = dest[109]
Image5 = dest[117]
Image6 = dest[252]
Image7 = dest[384]
Image8 = dest[457]
Image9 = dest[557]
Image10 = dest[668]
Image11 = dest[612]
Image12 = dest[705]
Image13 = dest[777]
Image14 = dest[791]
Image15 = dest[812]
Image16 = dest[859]
Image17 = dest[899]
Image18 = dest[907]
Image19 = dest[1072]
Image20 = dest[999]
Image21 = dest[951]
Image22 = dest[1077]
Image23 = dest[1000]
Image24 = dest[5]
Image25 = dest[10]
Image26 = dest[17]
Image27 = dest[33]
Image28 = dest[56]
Image29 = dest[78]
Image30 = dest[89]
Image31 = dest[1]
Image32 = dest[10]
Image33 = dest[81]
Image34 = dest[91]
Image35 = dest[1111]
Image36 = dest[443]


# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)

# showing image
plt.imshow(Image1)
plt.axis('off')

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)

# showing image
plt.imshow(Image2)
plt.axis('off')

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)

# showing image
plt.imshow(Image3)
plt.axis('off')

# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)

# showing image
plt.imshow(Image4)
plt.axis('off')

# Adds a subplot at the 5th position
fig.add_subplot(rows, columns, 5)

# showing image
plt.imshow(Image5)
plt.axis('off')

# Adds a subplot at the 6th position
fig.add_subplot(rows, columns, 6)

# showing image
plt.imshow(Image6)
plt.axis('off')

# Adds a subplot at the 7th position
fig.add_subplot(rows, columns, 7)

# showing image
plt.imshow(Image7)
plt.axis('off')

# Adds a subplot at the 8th position
fig.add_subplot(rows, columns, 8)

# showing image
plt.imshow(Image8)
plt.axis('off')

# Adds a subplot at the 9th position
fig.add_subplot(rows, columns, 9)

# showing image
plt.imshow(Image9)
plt.axis('off')

# Adds a subplot at the 10th position
fig.add_subplot(rows, columns, 10)

# showing image
plt.imshow(Image10)
plt.axis('off')

# Adds a subplot at the 11th position
fig.add_subplot(rows, columns, 11)

# showing image
plt.imshow(Image11)
plt.axis('off')

# Adds a subplot at the 12th position
fig.add_subplot(rows, columns, 12)

# showing image
plt.imshow(Image12)
plt.axis('off')

# Adds a subplot at the 13th position
fig.add_subplot(rows, columns, 13)

# showing image
plt.imshow(Image13)
plt.axis('off')

# Adds a subplot at the 14th position
fig.add_subplot(rows, columns, 14)

# showing image
plt.imshow(Image14)
plt.axis('off')

# Adds a subplot at the 15th position
fig.add_subplot(rows, columns, 15)

# showing image
plt.imshow(Image15)
plt.axis('off')

# Adds a subplot at the 16th position
fig.add_subplot(rows, columns, 16)

# showing image
plt.imshow(Image16)
plt.axis('off')

# Adds a subplot at the 17th position
fig.add_subplot(rows, columns, 17)

# showing image
plt.imshow(Image17)
plt.axis('off')

# Adds a subplot at the 18th position
fig.add_subplot(rows, columns, 18)

# showing image
plt.imshow(Image18)
plt.axis('off')

# Adds a subplot at the 19th position
fig.add_subplot(rows, columns, 19)

# showing image
plt.imshow(Image19)
plt.axis('off')

# Adds a subplot at the 20th position
fig.add_subplot(rows, columns, 20)

# showing image
plt.imshow(Image20)
plt.axis('off')

# Adds a subplot at the 21th position
fig.add_subplot(rows, columns, 21)

# showing image
plt.imshow(Image21)
plt.axis('off')

# Adds a subplot at the 22th position
fig.add_subplot(rows, columns, 22)

# showing image
plt.imshow(Image22)
plt.axis('off')

# Adds a subplot at the 23th position
fig.add_subplot(rows, columns, 23)

# showing image
plt.imshow(Image23)
plt.axis('off')

# Adds a subplot at the 24th position
fig.add_subplot(rows, columns, 24)

# showing image
plt.imshow(Image24)
plt.axis('off')

# Adds a subplot at the 25th position
fig.add_subplot(rows, columns, 25)

# showing image
plt.imshow(Image25)
plt.axis('off')

# Adds a subplot at the 26th position
fig.add_subplot(rows, columns, 26)

# showing image
plt.imshow(Image26)
plt.axis('off')

# Adds a subplot at the 27th position
fig.add_subplot(rows, columns, 27)

# showing image
plt.imshow(Image27)
plt.axis('off')

# Adds a subplot at the 28th position
fig.add_subplot(rows, columns, 28)

# showing image
plt.imshow(Image28)
plt.axis('off')

# Adds a subplot at the 29th position
fig.add_subplot(rows, columns, 29)

# showing image
plt.imshow(Image29)
plt.axis('off')


# Adds a subplot at the 30th position
fig.add_subplot(rows, columns, 30)

# showing image
plt.imshow(Image30)
plt.axis('off')

# Adds a subplot at the 31th position
fig.add_subplot(rows, columns, 31)

# showing image
plt.imshow(Image31)
plt.axis('off')

# Adds a subplot at the 32th position
fig.add_subplot(rows, columns, 32)

# showing image
plt.imshow(Image32)
plt.axis('off')

# Adds a subplot at the 33th position
fig.add_subplot(rows, columns, 33)

# showing image
plt.imshow(Image33)
plt.axis('off')

# Adds a subplot at the 34th position
fig.add_subplot(rows, columns, 34)

# showing image
plt.imshow(Image34)
plt.axis('off')

# Adds a subplot at the 35th position
fig.add_subplot(rows, columns, 35)

# showing image
plt.imshow(Image35)
plt.axis('off')

# Adds a subplot at the 36th position
fig.add_subplot(rows, columns, 36)

# showing image
plt.imshow(Image36)
plt.axis('off')
