random.seed(10) # For 5-shot learning

# Recall we have the training and valid splits, now we do the valid and test split.

shuffled = random.sample(files_list_miniImageNet,len(files_list_miniImageNet))
trainlist_final,_ = get_training_and_valid_sets(shuffled)
_,vallist = get_training_and_valid_sets(shuffled)

# For validation and test data splitting.

def get_valid_and_test_sets(file_list):
    split = 0.50           # 20 class set as test.
    split_index = floor(len(file_list) * split)
    # valid.
    training = file_list[:split_index]
    # test.
    validation = file_list[split_index:]
    return training, validation

validlist_final,_ = get_valid_and_test_sets(vallist)
_,testlist_final = get_valid_and_test_sets(vallist)

test_img = []

for test in testlist_final:
   data_test_img = load_images(path + '/' + test + '/')
   test_img.append(data_test_img)


############# test images + labels in array list format ##################

test_img_final = []
test_label_final = []

for e in range (len(test_img)):
   for f in range (600):   # Each class has 600 images.
      test_img_final.append(test_img[e][f])
      test_label_final.append(e+80)


############# Reassemble in tuple format. ##################

test_array = []

for e,f in zip(test_img_final,test_label_final):
  test_array.append((e,f))

################## shuffle #############################

test_array = shuffle(test_array)

new_X_test = [x[0] for x in test_array]
new_y_test = [x[1] for x in test_array]


test_dataset =  miniImageNet_CustomDataset(new_X_test,new_y_test, transform=[None])

################## One hot encode all label test array. ###########################

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# integer encode
label_encoder = LabelEncoder()
integer_encodedtest = label_encoder.fit_transform(new_y_test)


# binary encode. As of New ver, use sparse_output instead of sparse.
onehot_encodertest = OneHotEncoder(sparse_output = False)
integer_encodedtest= integer_encodedtest.reshape(len(integer_encodedtest), 1)
onehot_encodedtest= onehot_encodertest.fit_transform(integer_encodedtest)
print(np.shape(onehot_encodedtest))
print(len(onehot_encodedtest))


################# initialize our optimizer and model ############################
print("[INFO] compiling model...")
opt = Adam(1e-3)
model_test = SimpleCNN.build(width=84, height=84, depth=3, classes=18)
model_test.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
# train the simple CNN.
print("[INFO] training network...")
model_test.fit(np.asarray(new_X_test), np.asarray(onehot_encodedtest),
    batch_size=64,
    epochs=100,
    verbose=1)

#############################################################################################################

# now that our model is fine-tuned we should evaluate it on the test
# set (i.e., non-adversarial) again to see if performance has degraded
(loss, acc) = model_test.evaluate(x=np.asarray(new_X_test), y=np.asarray(onehot_encodedtest), verbose=0)
print("")
print("[INFO] normal testing images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))


################### generate a set of adversarial from our test set ##########################################
print("[INFO] generating adversarial examples with FGSM...\n")
(advXtest, advYtest) = next(generate_adversarial_batch(model_test, len(new_X_test),
    np.asarray(new_X_test), np.asarray(onehot_encodedtest), (84, 84, 3), eps=0.30))  # Changed the epsilon values here.
# re-evaluate the model on the adversarial images
(loss, acc) = model_test.evaluate(x=advXtest, y=advYtest, verbose=0)
print("[INFO] adversarial testing images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))

############################ Store test adversarial samples #################################################

test_array_adv = []

for A,B in zip(advXtest,onehot_encodertest.inverse_transform(advYtest)):  # Need include inverse_transform.
  test_array_adv.append((A,B))

Adv_X_test = [x[0] for x in test_array_adv]
Adv_y_test= [x[1] for x in test_array_adv]

Adv_intytest = []

for z in Adv_y_test:
     Adv_intytest.append(int(z))

test_dataset_adv =  miniImageNet_CustomDataset(Adv_X_test,Adv_intytest, transform=[None])

###########################################################################################

noisyItest= []

# Mean = 0, std = 0.005, 0.10, 0.15

for f1t in range (len(new_X_test)):
  imgt = new_X_test[f1t]
  mean = 0.0   # some constant
  std = 0.05   # some constant (standard deviation)
  noisy_imgIt = imgt + np.random.normal(mean, std, imgt.shape)
  noisy_img_clippedIt = np.clip(noisy_imgIt, 0, 255)  # we might get out of bounds due to noise
  noisy_img_clippedIt  = np.asarray(noisy_img_clippedIt) # REMEMBER TO ADD CONVERT TO ASARRAY FIRST BEFORE APPENDING!!!!!!
  noisyItest.append(noisy_img_clippedIt)

test_dataset_nat =  miniImageNet_CustomDataset(noisyItest, new_y_test, transform=[None])

##########################################################################################

test_dataloader = DataLoader([test_dataset,test_dataset_adv,test_dataset_nat], batch_size=16, shuffle=True)

############################### Evaluate on one task ########################################
def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels:torch.Tensor,
) -> [int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """
    return (
        torch.max(
            model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
            .detach()
            .data,
            1,
        )[1]
        == query_labels.cuda()
    ).sum().item(), len(query_labels)

def evaluate(data_loader: DataLoader):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    model.eval()
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):

            correct, total = evaluate_on_one_task(
                support_images.float(), support_labels, query_images.float(), query_labels
            )

            total_predictions += total
            correct_predictions += correct

    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
    )


################ Load test samplers and loaders code here. ####################################

# The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
test_dataset.get_labels = lambda: [instance[1] for instance in test_dataset]

test_sampler = TaskSampler(
    test_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)

test_loader = DataLoader(
    test_dataset,
    batch_sampler=test_sampler,
    num_workers=8,  # from 12.
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)


#################### Create support and query labels and images ###################

(example_support_images,
 example_support_labels,
 example_query_images,
 example_query_labels,
 example_class_ids,
) = next(iter(test_loader))

model.eval()
example_scores = model(
    example_support_images.cuda(),
    example_support_labels.cuda(),
    example_query_images.cuda(),
).detach()

_, example_predicted_labels = torch.max(example_scores.data, 1)
testlabels = [instance[1] for instance in test_dataset]

Eval = []

for i in range (10):
    E = evaluate(test_loader)
    Eval.append(E)
