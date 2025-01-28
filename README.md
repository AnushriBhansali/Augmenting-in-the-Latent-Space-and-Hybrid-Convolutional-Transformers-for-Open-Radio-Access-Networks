# Augmenting-in-the-Latent-Space-and-Hybrid-Convolutional-Transformers-for-Open-Radio-Access-Networks

## Overview  

This project explores **Open Radio Access Networks (O-RAN)**, leveraging **latent space augmentation** and **hybrid convolutional transformers** to enhance the performance of machine learning models in data-scarce scenarios. By implementing innovative data augmentation techniques, including **E-Mixup** and **E-Stitchup**, and benchmarking state-of-the-art computer vision models, this research pushes the boundaries of interference classification in O-RAN systems.

---

##  Key Features  

- **O-RAN Integration:**  
  - Focus on **Near-Real-Time RAN Intelligent Controllers (Near-RT RIC)** and spectrogram-based interference classification xApps.  
  - Utilizes shared RIC databases for centralized data handling.  

- **Latent Space Augmentation:**  
  - Implements advanced techniques like **E-Mixup** and **E-Stitchup** to create synthetic data in embedding space, enhancing model robustness and efficiency.  

- **Computer Vision Models Benchmarked:**  
  - Includes **CNNs** (ResNet, DenseNet, MobileNet), **Vision Transformers (ViT)**, and **Hybrid Architectures (ConvNeXt)** for spectrogram-based classification tasks.  

- **Performance Optimization:**  
  - Tackles challenges in noisy environments using **speckle noise** and evaluates model robustness.  

---

##  Results Summary  

### Noise-Free Dataset  
- **Best Model:** ResNet + E-Mixup  
  - **Accuracy:** 95.71%  
  - **F1-Score:** 0.957  

- **Other Notable Results:**  
  - MobileNet + E-Mixup: **94.67% Accuracy**  
  - DenseNet + E-Mixup: **94.95% Accuracy**  

### Noisy Dataset (Speckle Noise)  
- **Best Model:** MobileNet + E-Mixup  
  - **Accuracy:** 92.28%  
  - **F1-Score:** 0.922  

- **Other Notable Results:**  
  - ResNet + E-Stitchup: **91.33% Accuracy**  
  - DenseNet + E-Mixup: **87.71% Accuracy**  

---

##  Methodology  

### 1. **O-RAN Architecture:**  
   - Disaggregates traditional RAN components into open, modular functional blocks.  
   - xApps in Near-RT RIC leverage machine learning for interference classification.  

### 2. **Latent Space Augmentation:**  
   - **E-Mixup:** Interpolates embeddings to create synthetic data points.  
   - **E-Stitchup:** Combines embedding vectors to generate diverse class-specific samples.  

### 3. **Models Evaluated:**  
   - **CNNs:** ResNet, DenseNet, MobileNetV2  
   - **Vision Transformers (ViT):** Handles global image dependencies.  
   - **ConvNeXt:** Hybrid transformer-inspired CNN architecture.  

### 4. **Dataset:**  
   - **Total Samples:** 2100 spectrograms across three classes:  
     - Signal of Interest (SOI), Continuous Wave Interference (CWI), and Chirped Interference (CI).  
   - **Benchmark with Noise:** Added speckle noise for robustness evaluation.  

### 5. **t-SNE Visualizations:**  
   - Visualized feature separability under noise-free and noisy conditions to assess model robustness.

---

## 📂 Project Structure  

- **`Model-Notebooks/`**:  Implementations of CNNs, ViT, and ConvNeXt architectures.  
- **`Project paper Document/`**: Research, implementation and results documentation.  

---

## 🛠️ Setup and Usage  

### Prerequisites  
- Python >= 3.8  
- Required Libraries: `TensorFlow`, `PyTorch`, `scikit-learn`, `matplotlib`, `seaborn`, `numpy`  

### Installation  
1. Clone the repository:  
   ```bash
   git clone <repository_url>
   cd <repository_name>
