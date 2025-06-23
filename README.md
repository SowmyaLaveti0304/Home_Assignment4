# Home Assignment 4 – Summer 2025

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

<img width="655" alt="Screenshot 2025-06-23 at 2 54 18 PM" src="https://github.com/user-attachments/assets/cf420600-fd09-4154-9d80-9870a83aead5" />

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

---
### 4. The Adversarial Loop
- **Discriminator update:**
  - Freeze G.
  - Take a minibatch of real x and fake G(z), compute and step D’s parameters to maximize it.

- **Generator update:**
  - Freeze D.
  - Sample a fresh batch of z, generate G(z), compute and update G to minimize its loss (i.e.\ better fool D).

- **Repeat**
  - Each network continually adapts to the other:
  - D becomes a sharper detector of fakes.
  - G becomes ever more skilled at mimicking the real data distribution.
 
## Q2: Ethics and AI Harm  
### Misinformation in generative AI

#### Application Example:
An AI-powered news‐summarization tool that ingests live feeds and automatically writes breaking news articles.
- Harm: It may fabricate quotes, dates, or sources—misleading readers in real time.

---
Two Mitigation Strategies:

- Fact-Verification Pipeline:
  - After generation, automatically cross-check every named entity, quote, and statistic against a curated database of trusted outlets.
  - If any item fails verification, either flag the story for human review or redact the suspect passage until confirmed.

- Confidence-Threshold Abstention:
  - Train the model to emit a confidence score for each claim.
  - If confidence < threshold, the system emits a “Unverified” notice or defers to a human editor instead of publishing.

## Q3: simple GAN using TensorFlow to generate handwritten digits from the MNIST dataset 
## Q4: Simulate a data poisoning attack on a sentiment classifier
## Q5: Legal and Ethical Implications of GenAI
 **Legal & Ethical Concerns:**
#### Memorizing Private Data
 - Privacy Violation: Models that regurgitate verbatim snippets (names, addresses, even credit-card-like strings) risk exposing individuals’ personal info, breaching GDPR/CCPA.
 - Liability: Organizations can be held responsible if a deployed model leaks protected data.

#### Generating Copyrighted Material
- Infringement Risk: Producing large passages from works like Harry Potter may violate copyright law—even if unintentional—because it substitutes for licensed content.
- Fair Use Ambiguity: It’s unclear how “transformative” outputs must be to qualify as fair use, creating legal gray zones for developers and end-users.
---

**Should Training Data Be Restricted?**
Yes. At minimum, models should avoid ingesting non-public or un-licensed proprietary content:
 - Protect Privacy & Compliance: Excluding private communications and sensitive records prevents accidental leaks and eases regulatory compliance.
 - Respect IP Rights: Training only on public-domain or properly licensed corpora reduces infringement risk, fosters trust, and supports sustainable partnerships with rights holders.

## Q6: Bias & Fairness Tools
#### False Negative Rate Parity (Equal Opportunity)

- Metric measures: For each protected group, the share of real “positives” (e.g. qualified applicants) that the model wrongly labels negative.

- Why it matters: Ensures no group is systematically “missed” and denied opportunities they deserve.

- How it can fail: If Group A’s FNR is, say, 20% while Group B’s is 5%, qualified members of A are disproportionately rejected.
