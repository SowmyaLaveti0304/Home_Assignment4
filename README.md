# Home Assignment 3 – Summer 2025

**Student Name:** Sowmya Laveti  
**Student ID:** 700771347  
**University:** University of Central Missouri  
**Course:** CS5720 – Neural Networks and Deep Learning

---

## Assignment Overview

This assignment covers key topics in NLP and deep learning:

- **Q1:** GAN Architecture  
- **Q2:** Ethics and AI Harm  
- **Q3:** simple GAN using TensorFlow to generate handwritten digits from the MNIST dataset 
- **Q4:** Simulate a data poisoning attack on a sentiment classifier
- **Q5:** Legal and Ethical Implications of GenAI
- **Q6:** Bias & Fairness Tools

---
## Q1: GAN Architecture 
### Generator (G)
 - Takes random noise z and produces a “fake” sample G(z).
 - Goal: fool the Discriminator into thinking G(z) is real.
 - Training signal: gradients from the Discriminator’s classification loss.

---

### Discriminator (D)
 - Sees both real data x and fake data G(z).
 - Goal: assign D(x) ≈ 1D(x) for real and D(G(z)) ≈ 0 for fake.
 - Training signal: standard binary-cross-entropy loss on its real/fake decisions.

---
### Adversarial loop:
- Fix G, update D to better spot fakes.
- Fix D, update G to better fool D.
- Repeat—G gets more realistic, D gets more discerning—until G’s samples are (ideally) indistinguishable from real.

  ---
  
For the above GAN Architecture:
### 1. Inputs: Two Data Streams
#### Latent Noise z
Shown at bottom-left as a vertical stack of circles.
We sample Z ~ Pz(Z) (e.g. a standard normal vector).

#### Real Samples x
Shown at top-left as “Real world images” (the cylinder) → “Sample” box.
We draw x∼Pdata(x), our ground-truth data.

---
### 2. Generator (G)
- **Data flow:**
 		Z →G(Z)Discriminator input
- **Objective:**
 Generator  wants to generate samples that the Discriminator labels “real.”
- **Training signal:**
 The red arrow from the Discriminator back into G (“Differentiable module”) carries the gradient of D’s classification loss w.r.t. G’s parameters.
---
### 3. Discriminator (D)

- **Data flow:** Takes two inputs in parallel:
- Real sample 
 x → outputs D(x) (“Real” green light).

 - Fake sample 
G(z) → outputs D(G(z)) (“Fake” red light).

Both scores feed into the single “Loss” box on the right.
- **Objective:**
It wants D(x)→1 and D(G(z))→0.

- **Training signal:**
The red arrow from the “Loss” box back into D (“Differentiable module”) is the gradient used to update D’s weights.
