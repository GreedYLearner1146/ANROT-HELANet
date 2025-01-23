from torchvision.transforms import (
    Compose,
    RandomApply,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
)

from torchvision import transforms
from torchvision.transforms import v2
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import default_collate

################# We emphasize the role of data augmentation here. ##########################

# The data augmentation selected for the train and valid datasets are as below.

data_transform = transforms.Compose(

     [
            transforms.ToTensor(),
            transforms.Resize((84,84)),
            transforms.CenterCrop((84,84)),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([RandomRotation((90, 90))], p=0.5),
            transforms.RandomApply([RandomRotation((270, 270))], p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ])

data_transform_valtest = transforms.Compose(

      [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
       ])



class Augment:
   """
   A stochastic data augmentation module
   Transforms any given data example randomly
   resulting in two correlated views of the same example,
   denoted x ̃i and x ̃j, which we consider as a positive pair.
   """

   def __init__(self, img_size, s=1):
       color_jitter = T.ColorJitter(
           0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
       )
       # 10% of the image
       blur = T.GaussianBlur((3, 3), (0.1, 2.0))

       self.train_transform = torch.nn.Sequential(
           T.RandomResizedCrop(size=img_size),
           T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
           T.RandomApply([color_jitter], p=0.5),
           T.RandomApply([blur], p=0.5),
           T.RandomGrayscale(p=0.5), # 0.2
           # imagenet stats
           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       )

   def __call__(self, x):
       return self.train_transform(x), self.train_transform(x)


#################### Cutmix or MixUp data augmentation ################################

cutmix = v2.CutMix(num_classes=100)
mixup = v2.MixUp(num_classes=100)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

########################### miniImageNet dataloader ###################################

class miniImageNet_CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = data_transform
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        image = self.transform(np.array(image))
        return image, label
    def __len__(self):
        return len(self.labels)

#################################### Train and Valid Dataloader ##############################

train_dataset = miniImageNet_CustomDataset(new_X_train,new_y_train, transform=[data_transform, Augment]) # Combined data transform. Augment is from Data_Augmentation.py
val_dataset =  miniImageNet_CustomDataset(new_X_val,new_y_val, transform=[data_transform_valtest])
