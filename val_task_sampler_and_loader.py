# The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
val_dataset.get_labels = lambda: [instance[1] for instance in val_dataset]

val_sampler = TaskSampler(
    val_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS)

val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    num_workers=8,  # from 12.
    pin_memory=True,
    collate_fn=val_sampler.episodic_collate_fn,
)

#################### Create support and query labels and images ###################

(example_support_images,
 example_support_labels,
 example_query_images,
 example_query_labels,
 example_class_ids,
) = next(iter(val_loader))

model.eval()
example_scores = model(
    example_support_images.cuda(),
    example_support_labels.cuda(),
    example_query_images.cuda(),
).detach()

_, example_predicted_labels = torch.max(example_scores.data, 1)
vallabels = [instance[1] for instance in val_dataset]

############# You can plot some examples of support and query images using the two lines below ############
plot_images(example_support_images, "support images", images_per_row=N_SHOT)
plot_images(example_query_images, "query images", images_per_row=N_QUERY)
