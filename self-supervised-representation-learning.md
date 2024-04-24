---
description: >-
  There are currently 3 Self-Supervised encoder framework/types supported by
  SIT-FUSE for representation learning. The models range from small and compact
  to large and powerful:
---

# 🤖 Self-Supervised Representation Learning

* Restricted Boltzmann Machines (RBMs) + Deep Belief Networks (DBNs) - An RBM are simple 2-layer generative models, that are trained here using an unsupervised learning algorithm called contrastive divergence, during which, the optimizers goal is to be able to reconstruct samples that fall within the distribution of the input samples, X, by sampling the output of the model. In doing so, the model develops an in-depth understanding of the internal structure of the data.\
  RBMs can also be stacked together, to create a deep architecture called a Deep Belief Network (Hinton et al., 2006). Because of the nature of RBMs, the training process of DBNs can be done in a layer-wise fashion, making it less resource hungry than typical backpropagation-based model training techniques. SIT-FUSE allows for DBNs to be used as well as simple RBMs

<figure><img src=".gitbook/assets/image.png" alt=""><figcaption><p>A diagram depicting contrastive divergence</p></figcaption></figure>

<figure><img src=".gitbook/assets/Screenshot 2024-04-23 at 5.31.31 PM (1).png" alt=""><figcaption><p>A depiction of stacked RBMs, creating a DBN</p></figcaption></figure>



***

*   Bootstrap Your Own Latent - A framework that tries to minimize similarity loss between a pair of convolutional networks trained together, one seeing the unperturbed version of a sample, and the other seeing its perturbed pair

    <figure><img src=".gitbook/assets/Screenshot 2024-04-23 at 5.26.22 PM.png" alt=""><figcaption></figcaption></figure>

***

*   Image-Joint Embedding Predictive Architecture (I-JEPA) - A framework that trains a pair of Vision Transformers (ViTs) to learn a joint embedding by learning to predict values for masked out parts of an input tile.\


    <figure><img src=".gitbook/assets/Screenshot 2024-04-23 at 5.25.46 PM.png" alt=""><figcaption></figcaption></figure>



References:

1. Hinton, G. E. (2012). A Practical Guide to Training Restricted Boltzmann Machines.. In G. Montavon, G. B. Orr & K.-R. Müller (ed.), _Neural Networks: Tricks of the Trade (2nd ed.)_ , Vol. 7700 (pp. 599-619) . Springer . ISBN: 978-3-642-35288-1.
2. Carreira-Perpinan, M. A. & Hinton, G. E. (2005). On Contrastive Divergence Learning . In Intelligence, A. & Statistics, 2005, B. (ed.), .\

3. Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko: “Bootstrap your own latent: A new approach to self-supervised Learning”, 2020; [arXiv:2006.07733](http://arxiv.org/abs/2006.07733).
4. Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas: “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture”, 2023; [arXiv:2301.08243](http://arxiv.org/abs/2301.08243).
5. Renjie Liao, Simon Kornblith, Mengye Ren, David J. Fleet, Geoffrey Hinton: “Gaussian-Bernoulli RBMs Without Tears”, 2022; [arXiv:2210.10318](http://arxiv.org/abs/2210.10318).
