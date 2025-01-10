# ANROT-HELANet

**This code repository is still being updated in progress.** 

This is the code repository for our ANROT-HELANet, an extension of our previous HELA-VFA work to be robust to adversarial and natural gaussian noise attacks.

Code repository for HELA-VFA: https://github.com/GreedYLearner1146/HELA-VFA/tree/main 

We utilized the Sicara few-shot library package for running our few-shot algorithm. \
Link to the Sicara Few-Shot github page: https://github.com/sicara/easy-few-shot-learning.

All codes that is and will be shown here are presented in PyTorch format. The backbone for our ANROT-HELANet, like the HELA-VFA, is the ResNet-12.

**(This repo is still under constructed currently. Stay tune for changes.)**

# Abstract 

Few-Shot Learning (FSL) models have shown promising performance in learning to generalize from limited data samples, outperforming conventional neural network architectures like Convolutional Neural Networks (CNNs) that require large datasets. To further improve FSL, recent Bayesian estimation approaches have been introduced which help mitigate challenges faced by feature point methods. These Bayesian methods often use the Kullback-Leibler (KL) divergence for comparing probability distributions. However, both strategies remain vulnerable to adversarial and natural image noises. We present an adversarially and naturally robust FSL model called ANROT-HELANet, which utilizes Hellinger attention aggregation. Inspired by prototypical networks, ANROT-HELANet implements a robust Hellinger distance-based feature class aggregation scheme. We also introduce a Hellinger similarity contrastive loss function, which extends the cosine similarity loss to effectively deal with variational FSL. ANROT-HELANet demonstrates adversarial and natural noise robustness using an alternate f-divergence measure, emphasizing attention in the encoder to improve classification. Comprehensive benchmark experiments have confirmed the effectiveness of ANROT-HELANet. The new loss significantly enhances performance when utilized alongside categorical cross-entropy. We also explore the robustness of ANROT-HELANet against perturbations and evaluate the quality of its reconstructed images.


