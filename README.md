# Hand X-ray Severity Prediction using ResNet34 + CBAM Attention

##  Overview
This project focuses on predicting a continuous clinical severity score from 2D hand and wrist X-ray images using deep learning.

Unlike classification tasks, this regression problem requires the model to capture subtle radiographic patterns such as joint spacing, bone erosion, density variations, and structural asymmetry.

To address this, an enhanced deep learning model was developed using a pretrained ResNet34 backbone, CBAM attention modules, and multi-scale feature fusion. The model significantly improves performance over a baseline CNN.

---

##  Objectives
- Predict continuous severity scores from X-ray images  
- Capture fine-grained radiographic features  
- Improve performance over baseline CNN  
- Apply attention mechanisms for better feature focus  
- Optimize training for stable convergence  

---

##  Model Architecture

### Baseline Model
- SimpleCNN (provided starter model)  
- Limited depth and no pretrained features  

### Proposed Student Model
- ResNet34 pretrained on ImageNet  
- CBAM attention (Channel + Spatial attention)  
- Multi-scale feature fusion (Layer3 + Layer4)  
- Fully connected regression head  

---

##  Key Features
- Pretrained ResNet34 for strong feature extraction  
- CBAM attention to focus on important regions and features  
- Multi-scale feature fusion to combine local and global patterns  
- Regularization using BatchNorm, Dropout, and Weight Decay  
- Optimized training using AdamW and Cosine Learning Rate Scheduler  

---

##  Dataset
- Hand/Wrist X-ray images  
- Continuous severity score labels  
- Small dataset with high variability  

Note: Dataset is not included due to size constraints.

---

##  Results

### Baseline (SimpleCNN)
- MAE: 26.85  
- RMSE: 34.01  
- R²: 0.4336  

### Final Model (Optimized Student Model)
- MAE: 16.82  
- RMSE: 23.07  
- R²: 0.7392  

### Improvements
- MAE reduced by 37%  
- RMSE reduced by 32%  
- R² improved by +0.30  

---

##  Training Configuration
- Optimizer: AdamW  
- Learning Rate: 5e-4  
- Scheduler: Cosine Annealing  
- Batch Size: 16  
- Weight Decay: 1e-5  
- Epochs: 50  

---

##  Inference
- Predicts continuous severity scores  
- Captures overall trends effectively  
- Slight challenges with extreme values (common in regression tasks)  

---

## How to Run

### Install dependencies
``` bash
pip install -r requirements.txt
```

### Train model
``` bash
python main.py --model student --epochs 50 --batch_size 16 --lr 5e-4
```

### Run inference
``` bash
python inference.py
```

---

##  Tech Stack
- Python  
- PyTorch  
- NumPy  
- OpenCV  

---

## Key Contributions
- Designed an advanced regression model using ResNet34 and CBAM  
- Implemented multi-scale feature fusion  
- Improved performance significantly over baseline  
- Built a complete training and evaluation pipeline  

---

##  Project Report
Detailed explanation is included in the repository.

---

##  Google Colab
https://colab.research.google.com/drive/1oBt1ItjRVAXTvO9Bg08UzW_aHDoGaaus?usp=sharing

---

##  Academic Use & References
This project was developed as part of a graduate-level coursework in image processing and deep learning.

The implementation is inspired by:
- ResNet architectures  
- CBAM attention mechanisms  
- Deep learning techniques for medical imaging  

This work is intended for educational and research purposes.

---

## 📄 License
This project is for academic use only.
