# 🧠 ICLR 2026 — Transfer Learning & Domain Generalization Summary

## 🌍 Key Research Directions (High-Level Topics)

1. RL Transfer (Sim-to-Real)
   - Transfer reinforcement learning (RL) agents from simulation to the real world.
2. Text-to-Motion 3D Generation
   - Text-conditioned 3D or 4D motion synthesis.
3. Test-Time Adaptation & Scaling
   - Test-time training, optimization, compute scaling, and alignment.
4. 4D World Modeling
   - Multi-domain, multi-modal 4D world modeling and dynamics.
5. Foundation Model Transfer
   - Applying pre-trained foundation models to specialized downstream tasks.

------

## 🧩 Domain Generalization & Adaptation

### **1. Reasoning-Driven MLLMs**

**Paper:** *Reasoning-Driven Multimodal LLM for Domain Generalization*

- Highlight: Construct reasoning chains to derive image categories.

### **2. Mode Connectivity**

**Paper:** *Exploring Mode Connectivity in Krylov Subspace for Domain Generalization*

- Idea: Leverages local flatness around minima for better generalization.

### **3. PAS (Potential Adaptability Score)**

**Paper:** *Estimating the Target Accuracy before Domain Adaptation*

- Contribution: A pre-adaptation metric to estimate transferability between source and target domains.

### **4. Bayesian Prototype Evolution**

**Paper:** *Bayesian Evidence-Driven Prototype Evolution for Federated Domain Adaptation*

- Key: Dynamic evolution of prototype topology guided by Bayesian evidence.

### **5. Domain Unlearning**

**Paper:** *Unlearning during Training: Domain-Specific Gradient Ascent*

- Idea: Identify and remove samples that harm generalization.

------

## ⚙️ Hallucination Detection & Unlearning

### **Beyond In-Domain Detection: SpikeScore**

- Defines a “SpikeScore” for detecting hallucination based on multi-turn dialogue fluctuations.

### **Selective Data Removal**

- Finds key samples that encode most of a domain’s statistical influence.

### **VeriTrail**

- Closed-domain hallucination detection with traceability (for both MGS & SGS).

------

## 🧠 Representation & Similarity Learning

### **ELViS**

- *Efficient Visual Similarity from Local Descriptors*
- Works in **similarity space** rather than representation space.

### **Differentially Private Domain Discovery**

- Proposes *Weighted Gaussian Mechanism (WGM)* with near-optimal missing mass guarantees.

### **Elastic Optimal Transport**

- Adaptive-mass preserving transport formulation with theoretical guarantees.

------

## 🪞 Multi-Domain Image & Video Tasks

### **AiOIR**

- *Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration.*

### **NAG (Noise-Aware Generalization)**

- Jointly studies robustness to in-domain noise and out-of-domain generalization.

### **Universal Multi-Domain Translation**

- Diffusion routers handle domain translation with a single unified model.

### **VUDG Dataset**

- Video dataset focused on domain generalization with progressive expert annotation.

------

## 🏥 Biomedical & Robust Learning

### **Adaptive Test-Time Training (TTT) for ICU**

- Predicting mechanical ventilation needs with EHR-based test-time training.

### **Distributionally Robust Classification**

- Models uncertainty in both covariates and label distributions.

------

## 🧮 Optimization, Theory & Robust RL

### **Minimax-Optimal Aggregation for DRE**

- Ensemble-based method for density ratio estimation.

### **Cross-Domain Policy Optimization**

- Uses Bellman consistency and hybrid critics for cross-domain RL.

### **Dual-Robust Cross-Domain Offline RL**

- Dual robustness to dynamic shifts during training and testing.

### **Latent Adaptation of Foundation Policies**

- Sim-to-real transfer via latent foundation policy adaptation.

------

## 🧭 Planning, Reasoning & LLM Adaptation

### **CoPiC: Code Driven Planning**

- Uses LLMs to generate code-programs for planning tasks with a **domain-adaptive selector**.

### **CGPO**

- *Curvature-Guided Policy Optimization* for LLM reasoning across domains.

### **Foundational Automatic Evaluators**

- Large-scale generative evaluators for reasoning-centric tasks.

### **Test-Time Alignment for LLMs**

- Uses *Textual Model Predictive Control* for test-time adaptation.

------

## 🧠 Graphs, Causality & Structure

### **Learning Structure-Semantic Evolution Trajectories**

- Uses stochastic differential equations to model graph evolution.

### **TRIDENT**

- Cross-domain trajectory representation preserving spatio-temporal distances.

### **TriC-Motion**

- *Tri-Domain Causal Text-to-Motion Generation* via diffusion with causal interventions.

### **Frequency vs. Time-Domain Causal Recovery**

- Shows frequency-domain (FFT-based) methods outperform time-domain methods in structure recovery.

------

## 🧪 Diffusion, Generative & Cross-Modality Models

### **Adaptive Domain Shift in Diffusion Models**

- Embeds domain shift dynamics directly in the generative process.

### **Mirror Flow Matching**

- Incorporates heavy-tailed priors for convex domain modeling.

------

## 📊 Evaluation & Creativity

### **Text Creativity Evaluation**

- Proposes a dataset and pairwise comparison framework for cross-domain creativity judgment.

------

## 🧩 Test-Time Adaptation (TTT/TTA) — Focus Cluster

**Emerging themes:**

- Scaling, memorization, compute-efficiency, mixture of models, bioactivity prediction, and domain shifts.

**Representative works:**

1. **NEO:** No-Optimization TTA (latent re-centering).
2. **When & Where to Reset Matters:** Insights on long-term TTA.
3. **COSA:** Context-aware output adapter (time-series forecasting).
4. **EvoTest:** Evolutionary TTT for self-improving agentic systems.
5. **VITA:** Test-time adaptation of VLMs into zero-shot value functions.
6. **CaTS / ATTS:** Calibrated and asynchronous test-time scaling.
7. **Flatness-Guided TTA** for vision-language models.
8. **Bilateral Information-aware TTA:** Balances modality contributions.
9. **Mode-conditioning for compute scaling.**
10. **TTOM:** Test-time optimization & memorization (for compositional video generation).

------

## 🧰 Miscellaneous Novel Directions

- **Domain Expansion:** Latent space restructuring for multi-task learning.
- **Cross-Domain Lossy Compression:** OT-based tradeoff between rate and classification accuracy.
- **GOOD:** Geometry-guided out-of-distribution (OOD) test-time adaptation for 3D point clouds.
- **TTT3R:** 3D Reconstruction as test-time training.
- **VFScale:** Verifier-free reasoning scaling in diffusion models.
- **Strategic Bandit-based Compute Scaling.**

------

## 📈 Summary of Trends

| Theme                               | Direction                              | Example Keywords                     |
| ----------------------------------- | -------------------------------------- | ------------------------------------ |
| **Test-Time Adaptation & Scaling**  | Compute efficiency, memory, alignment  | TTT, scaling, alignment              |
| **Foundation Model Transfer**       | Task-specific adaptation               | LLM, VLM, diffusion, policy transfer |
| **Robustness & Causality**          | Noise, drift, structure                | Causal, noise-aware, dual-robust     |
| **Generative & Diffusion Models**   | Cross-domain synthesis                 | TriC-Motion, Adaptive Diffusion      |
| **Hallucination & Trustworthiness** | Detection, traceability                | SpikeScore, VeriTrail                |
| **Video & Graph DG**                | Temporal and structural generalization | TRIDENT, VUDG                        |
| **Evaluation Techniques**           | Creativity, domain influence           | PAS, Text Creativity Evaluator       |

