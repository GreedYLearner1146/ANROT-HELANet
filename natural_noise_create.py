# Add gaussian noise.

noisyI = []

for f1 in range (len(new_X_train)):
  img = new_X_train[f1]
  mean = 0.0   # some constant
  std = 0.05   # some constant (standard deviation). Can change values here.
  noisy_imgI = img + np.random.normal(mean, std, img.shape)
  noisy_img_clippedI = np.clip(noisy_imgI, 0, 255)  # we might get out of bounds due to noise
  noisy_img_clippedI  = np.asarray(noisy_img_clippedI) # REMEMBER TO ADD CONVERT TO ASARRAY FIRST BEFORE APPENDING!!!!!!
  noisyI.append(noisy_img_clippedI)


noisyIval = []


for f1v in range (len(new_X_val)):
  imgv = new_X_val[f1v]
  mean = 0.0   # some constant
  std = 0.05   # some constant (standard deviation). Can change values here.
  noisy_imgIv = imgv + np.random.normal(mean, std, imgv.shape)
  noisy_img_clippedIv = np.clip(noisy_imgIv, 0, 255)  # we might get out of bounds due to noise
  noisy_img_clippedIv  = np.asarray(noisy_img_clippedIv) # REMEMBER TO ADD CONVERT TO ASARRAY FIRST BEFORE APPENDING!!!!!!
  noisyIval.append(noisy_img_clippedIv)

#################################### Dataloader ##############################

train_dataset_natural_noise = miniImageNet_CustomDataset(noisyI,new_y_train, transform=[None]) # Combined data transform. Augment is from Data_Augmentation.py
val_dataset_natural_noise =  miniImageNet_CustomDataset(noisyIval,new_y_val, transform=[None])
