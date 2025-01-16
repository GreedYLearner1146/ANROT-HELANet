
#################################### Dataloader ##############################   transform=[data_transform, Augment], [data_transform_valtest]

train_dataset = miniImageNet_CustomDataset(Adv_X_train,Adv_intytrain, transform=[None]) # Combined data transform. Augment is from Data_Augmentation.py
val_dataset =  miniImageNet_CustomDataset(Adv_X_val, Adv_intyval , transform=[None])

train_dataloader= DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn) # Collate_fn called on here.
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True) # Collate_fn called on here.
