# ANROT-HELANet

**This code repository is still being updated in progress.** 

This is the code repository for our ANROT-HELANet, an extension of our previous HELA-VFA work to be robust to adversarial and natural gaussian noise attacks. The work has been submitted and is currently under revision for Elsevier's International Journal of Multimedia Information Retrieval.

Code repository for HELA-VFA: https://github.com/GreedYLearner1146/HELA-VFA/tree/main 

We utilized the Sicara few-shot library package for running our few-shot algorithm. \
Link to the Sicara Few-Shot github page: https://github.com/sicara/easy-few-shot-learning.

All codes that is and will be shown here are presented in PyTorch format. The backbone for our ANROT-HELANet, like the HELA-VFA, is the ResNet-12.

**(This repo is still under construction currently. Stay tune for changes.)**

# Abstract 

Few-Shot Learning (FSL), which involves learning to generalize using only a few data samples, has demonstrated promising and superior performances to ordinary CNN methods. While Bayesian based estimation approaches using Kullback-Leibler (KL) divergence have shown improvements, they remain vulnerable to adversarial attacks and natural noises. We introduce ANROT-HELANet, an Adversarially and Naturally RObusT Hellinger Aggregation Network that significantly advances the state-of-the-art in FSL robustness and performance. Our approach implements an adversarially and naturally robust Hellinger distance-based feature class aggregation scheme, demonstrating resilience to adversarial perturbations up to $\epsilon=0.30$ and Gaussian noise up to $\sigma=0.30$. The network achieves substantial improvements across benchmark datasets, including gains of 1.20\% and 1.40\% for 1-shot and 5-shot scenarios on miniImageNet respectively. We introduce a novel Hellinger Similarity contrastive loss function that generalizes cosine similarity contrastive loss for variational few-shot inference scenarios. Our approach also achieves superior image reconstruction quality with a FID score of 2.75, outperforming traditional VAE (3.43) and WAE (3.38) approaches. Extensive experiments conducted on four few-shot benchmarked datasets verify that ANROT-HELANet's combination of Hellinger distance-based feature aggregation, attention mechanisms, and our novel loss function establishes new state-of-the-art performance while maintaining robustness against both adversarial and natural perturbations. 

The preprint for our work can be found at https://arxiv.org/abs/2509.11220. 

# Model Architecture Figures

The attention mechanism as well as the adversarial samples creation via FGSM:
![ANROT_HELANET_1](https://github.com/user-attachments/assets/29bc3746-1741-4942-88fc-638ea2bebe8d)

Model Architecture
![Art_HELANET_structure](https://github.com/user-attachments/assets/476ba317-1f74-4308-ae43-bf17e28d1692)

# Preliminary Results

The selected methods (to be appeared in the paper to be published) are based on the approaches by Roy et.al. `FeLMi : Few shot Learning with hard Mixup' [1]. The few-shot evaluation utilized are the 5-way-1-shot and 5-way-5-shot approach. The table below tabulated the relevant values obtained for the miniImageNet benchmarked dataset when using the 5-way-1-shot and 5-way-5-shot approaches (in %):

| Method | 5-way-1-shot (%) | 5-way-5-shot (%) |
| ------ | ------| ------| 
|ANROT-HELANet| **69.4 $\pm$ 0.3** | **88.1 $\pm$ 0.4** |

The instructional code and values for the remaining benchmarked will be available in the future. The backbone for our model here is the ResNet-12.

# Image Reconstruction Preliminary Results

![image](https://github.com/user-attachments/assets/aebcc368-3695-41a0-8fc7-87619611fc50)

We note that in our experiment in this section, the Frechet Inception Distance (FID) [2] was used to evaluate the quality of the reconstructed images from our generative model. The smaller the FID, the better the reconstructed image quality. We utilized the miniImageNet images for the reconstruction evaluation.

| Method | FID |
| ------ | ------|
|*KL Divergence**| 3.43 |
|*Wasserstein Distance**|3.38 |
|*Hellinger Distance**| **2.75** |


## Code Instructions (for the adversarial and natural noise robustness training) ##
The codes instructions presented in this github utilized miniImageNet as an example. The instructional codes for the remaining benchmarked will be available in the future.

1) Run data_loading.py which load the datasets of your choice. Here we load and run the miniImageNet dataset.
2) Run data_augmentation_and_dataloader.py., which contains the data augmentation procedure and the pytorch dataloader class for the meta-train and valid set.
3) Run one_hot_encode.py to one-hot encode the labels of the datasets.
4) Run FGSM_adv_create.py which creates the adversarial samples and store them in the respective arrays using the FGSM [3].
5) Run natural_noise_create.py which creates the image corrupted by gaussian noise and store them in the respective arrays.
6) Run the overall_dataloader.py to load the dataloader that comprised of the original images, adversarially corrupted images, and the gaussian corrupted images, all along with their respective labels. This facilitates the simultaneous adversarial and natural noise training to enhance their robustness to such noises in the test phase.
7) Run Attention.py and ResNet12.py, which are part of the encoder component of the ANROT-HELANet.
8) Run Hellinger_dist.py, which contains the hellinger distance computation code.
9) Run ANROT_HELANet.py which contains the main code backbone of our algorithm.
10) Run hyperparameters.py, followed by val_task_sampler_and_loader.py and train_task_sampler_and_loader.py.
11) Run the sub-functions contained in the folder Hesim, which comprise the codes for the various helper functions leading up to the Hesim loss function as highlighted in our paper. The helper functions are mainly adapted from the pytorch metric learning library by Kevin Musgrave: https://github.com/KevinMusgrave/pytorch-metric-learning. Please run the functions in the following orders: Common_functions.py -> loss_and_miners_utils.py -> Module_With_Records.py -> Base_Reducers.py -> MeanReducer.py -> MultipleReducers_Do_Nothing_Reducers.py -> BaseDistances.py -> LpDistance.py -> ModulesWithRecordsandReducer.py -> Mixins.py -> BaseMetricLossFunction.py -> GenericPairLoss.py -> HesimLoss.py.
12) Run loss_functions.py, which contains the categorical cross-entropy loss and Hesim loss code combination.
13) Run training.py for the training loop.
14) Finally, run test_evaluation.py for the evaluation of our model on the meta-test dataset.

## Code Instructions (for generating the reconstructed images) ##

1) Run FID.py, which contains the code for running the Frechet Inception Distance computation for the first 10000 miniImageNet samples to be reconstructed.
2) Run reconstruction.py, which contains the backbone code for running the image reconstruction and FID computation. In the VAE class, the various distances laid out in the paper can be computed and compared via un-commenting and commenting the respective distance code sections (KL divergence, Hellinger Distance, Wasserstein Distance).

## Citation Information ##

If you find our work useful, please cite them as below: 

Lee, G.Y., Dam, T., Ferdaus, M.M., Poenar, D.P., Duong, V.N.: Anrot-helanet: Adverserially and naturally robust attention-based aggregation network via the hellinger distance for few-shot classification. arXiv preprint arXiv:2509.11220 (2025).

Alternatively, you can use the following BibTeX entry:

@article{lee2025anrot, \
  title={ANROT-HELANet: Adverserially and Naturally Robust Attention-Based Aggregation Network via The Hellinger Distance for Few-Shot Classification}, \
  author={Lee, Gao Yu and Dam, Tanmoy and Ferdaus, Md Meftahul and Poenar, Daniel Puiu and Duong, Vu N}, \
  journal={arXiv preprint arXiv:2509.11220}, \
  year={2025}
}


## Relevant References ##

[1] A. Roy, A. Shah, K. Shah, P. Dhar, A. Cherian, and R. Chellappa, “Felmi: Few shot learning with hard mixup,” in Advances in Neural Information Processing Systems, 2022. 5, 6. \
[2]  M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” Advances in neural information processing systems, vol. 30, 2017. \
[3] I. J. Goodfellow, J. Shlens, C. Szegedy, Explaining and harnessing adversarial examples, arXiv preprint arXiv:1412.6572 (2014).
