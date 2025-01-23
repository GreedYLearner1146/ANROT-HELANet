
#################################### Dataloader ##############################   transform=[data_transform, Augment], [data_transform_valtest]

train_dataloader= DataLoader([train_dataset,train_dataset_adv,train_dataset_nat], batch_size=16, shuffle=True, collate_fn=collate_fn) # Collate_fn called on here.
val_dataloader = DataLoader([val_dataset,val_dataset_adv,val_dataset_nat], batch_size=16, shuffle=True) # Collate_fn called on here.
