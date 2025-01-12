# ANROT-HELANet

**This code repository is still being updated in progress.** 

This is the code repository for our ANROT-HELANet, an extension of our previous HELA-VFA work to be robust to adversarial and natural gaussian noise attacks. The work has been submitted and is currently under revision for Elsevier's International Journal of Multimedia Information Retrieval.

Code repository for HELA-VFA: https://github.com/GreedYLearner1146/HELA-VFA/tree/main 

We utilized the Sicara few-shot library package for running our few-shot algorithm. \
Link to the Sicara Few-Shot github page: https://github.com/sicara/easy-few-shot-learning.

All codes that is and will be shown here are presented in PyTorch format. The backbone for our ANROT-HELANet, like the HELA-VFA, is the ResNet-12.

**(This repo is still under construction currently. Stay tune for changes.)**

# Abstract 

Few-Shot Learning (FSL) models have shown promising performance in learning to generalize from limited data samples, outperforming conventional neural network architectures like Convolutional Neural Networks (CNNs) that require large datasets. To further improve FSL, recent Bayesian estimation approaches have been introduced which help mitigate challenges faced by feature point methods. These Bayesian methods often use the Kullback-Leibler (KL) divergence for comparing probability distributions. However, both strategies remain vulnerable to adversarial and natural image noises. We present an adversarially and naturally robust FSL model called ANROT-HELANet, which utilizes Hellinger attention aggregation. Inspired by prototypical networks, ANROT-HELANet implements a robust Hellinger distance-based feature class aggregation scheme. We also introduce a Hellinger similarity contrastive loss function, which extends the cosine similarity loss to effectively deal with variational FSL. ANROT-HELANet demonstrates adversarial and natural noise robustness using an alternate f-divergence measure, emphasizing attention in the encoder to improve classification. Comprehensive benchmark experiments have confirmed the effectiveness of ANROT-HELANet. The new loss significantly enhances performance when utilized alongside categorical cross-entropy. We also explore the robustness of ANROT-HELANet against perturbations and evaluate the quality of its reconstructed images.

# Model Architecture Figures

The attention mechanism as well as the adversarial samples creation via FGSM:
![ANROT_HELANET_1](https://github.com/user-attachments/assets/29bc3746-1741-4942-88fc-638ea2bebe8d)

Model Architecture
![Art_HELANET_structure](https://github.com/user-attachments/assets/476ba317-1f74-4308-ae43-bf17e28d1692)

# Preliminary Results

The selected methods (to be appeared in the paper to be published) are based on the approaches by Roy et.al. `FeLMi : Few shot Learning with hard Mixup' [1]. The few-shot evaluation utilized are the 5-way-1-shot and 5-way-5-shot approach. The table below tabulated the relevant values obtained for each benchmarked datasets when using the 5-way-1-shot and 5-way-5-shot approaches (in %):

| Method | 5-way-1-shot (%) | 5-way-5-shot (%) |
| ------ | ------| ------| 
|*CIFAR-FS**| **78.7 $\pm$ 0.7** | **90.6 $\pm$ 0.5** |
|*FC-100**| **50.2 $\pm$ 0.6** | **69.6 $\pm$ 0.6** |
|*miniImageNet**| **69.4 $\pm$ 0.3** | **88.1 $\pm$ 0.4** |
|*tiredImageNet**| **73.5 $\pm$ 0.2** | **88.5 $\pm$ 0.8** |

The backbone for our HELA-VFA is the ResNet-12.

# Image Reconstruction Preliminary Results

![image](https://github.com/user-attachments/assets/aebcc368-3695-41a0-8fc7-87619611fc50)

We note that in our experiment in this section, the Frechet Inception Distance (FID) [2] was used to evaluate the quality of the reconstructed images from our generative model. The smaller the FID, the better the reconstructed image quality.

| Method | FID |
| ------ | ------|
|*KL Divergence**| 3.43 |
|*Wasserstein Distance**|3.38 |
|*Hellinger Distance**| **2.75** |


## Code Instructions ##
The codes instructions presented in this github utilized miniImageNet as an example.
