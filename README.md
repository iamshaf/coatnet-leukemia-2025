Bridging AI and Hematology: A Novel
Hybrid Model for Accurate Blood Cancer
Type Classification
Overview
This repository contains the implementation of a CoatNet (Convolution + Attention) model for
automated classification of blood cell types, specifically targeting leukemia detection. The
model combines convolutional neural networks with transformer attention mechanisms to
achieve high accuracy in distinguishing between different types of white blood cells.
Abstract
This work presents a novel application of the CoatNet architecture for medical image
classification, specifically for identifying different types of blood cells that are crucial in
leukemia diagnosis. The hybrid model leverages both convolutional layers for local feature
extraction and transformer blocks for global context understanding, resulting in improved
classification performance compared to traditional CNN approaches.
Dataset
The model is trained and evaluated on a blood cell dataset containing four distinct classes:
• EOSINOPHIL: A type of white blood cell involved in allergic reactions
• LYMPHOCYTE: Key immune system cells, abnormal counts may indicate leukemia
• MONOCYTE: Large white blood cells that fight infections
• NEUTROPHIL: Most abundant white blood cells, first responders to infections
Dataset Structure
dataset2-master/
├── images/
│ ├── TRAIN/
│ │ ├── EOSINOPHIL/
│ │ ├── LYMPHOCYTE/
│ │ ├── MONOCYTE/
│ │ └── NEUTROPHIL/
│ └── TEST_SIMPLE/
│ ├── EOSINOPHIL/
│ ├── LYMPHOCYTE/
│ ├── MONOCYTE/
│ └── NEUTROPHIL/
Model Architecture
CoatNet Hybrid Architecture
The implemented CoatNet model combines:
1. Convolutional Stages (Stages 0-1):
o MobileNet-style inverted bottleneck convolutions (MBConv)
o Efficient feature extraction with depthwise separable convolutions
o SiLU activation functions for improved gradient flow
2. Transformer Stages (Stages 2-4):
o Multi-head self-attention mechanisms
o Relative positional encoding
o GELU activation in feed-forward networks
o Layer normalization and residual connections
Key Components
Custom Layers
• SiLUActivation: Sigmoid Linear Unit activation
• GELUActivation: Gaussian Error Linear Unit activation
• RelativeAttention: Custom attention mechanism with relative positioning
Architecture Specifications
• Input Size: 224×224×3 RGB images
• Stages: [2, 2, 3, 5, 2] blocks per stage
• Channels: [64, 96, 192, 384, 768] progressive channel expansion
• Attention Heads: [1, 1, 2, 4, 8] multi-head attention configuration
Implementation Details
Preprocessing
• Images resized to 224×224 pixels
• MobileNetV2 preprocessing applied
• Data augmentation during training
Training Configuration
• Optimizer: AdamW with weight decay (1e-5)
• Learning Rate: 1e-4
• Loss Function: Sparse Categorical Crossentropy
• Batch Size: 32
• Data Split: 68% train, 17% validation, 15% test
Key Features
• Mixed Precision Training: GPU optimization
• Dropout Regularization: 0.1 in attention, 0.2 before classification
• Batch Normalization: Stability in convolutional layers
• Global Average Pooling: Dimensional reduction before classification
Results and Evaluation
The model is evaluated using comprehensive metrics:
Performance Metrics
• Accuracy: Overall classification accuracy
• Precision, Recall, F1-Score: Per-class performance metrics
• Confusion Matrix: Detailed classification breakdown
• ROC Curves: Area Under Curve (AUC) analysis
Visualization Tools
• Training/validation loss and accuracy curves
• Confusion matrix heatmaps
• Sample prediction visualizations (correct/incorrect)
• Probability distribution analysis
• ROC curve plots for multi-class classification
File Structure
project/
├── cs-119389-coatnet-leukemia-paper-2024_(1).ipynb
├── training_history.csv (generated)
├── model_evaluation_results.txt (generated)
└── README.md
Requirements
Dependencies
tensorflow>=2.8.0
numpy
pandas
scikit-learn
matplotlib
seaborn
Hardware Requirements
• GPU recommended for training (tested on Kaggle GPU environment)
• Minimum 8GB RAM
• CUDA-compatible GPU for optimal performance
Usage
Training the Model
1. Prepare the dataset in the specified directory structure
2. Run the data preprocessing cells
3. Execute the model definition and compilation
4. Train the model using the provided data generators
Evaluation
The notebook includes comprehensive evaluation tools:
• Model performance on test set
• Detailed classification reports
• Visual analysis of predictions
• Training history visualization
Clinical Significance
This work contributes to automated medical diagnosis by:
• Reducing Manual Labor: Automated blood cell classification
• Improving Accuracy: Hybrid architecture for better feature learning
• Early Detection: Potential for early leukemia screening
• Standardization: Consistent classification across different laboratories
Future Work
Potential improvements and extensions:
• Data Augmentation: Advanced augmentation techniques
• Ensemble Methods: Combining multiple model predictions
• Explainability: Attention visualization and GradCAM analysis
• Clinical Validation: Testing on larger, diverse clinical datasets
• Real-time Implementation: Mobile/edge device deployment
Citation
If you use this work in your research, please cite:
@article{coatnet_leukemia_2024,
 title={CoatNet for Blood Cell Classification: A Deep Learning Approach
to Leukemia Detection},
 author={[Your Name]},
 journal={[Journal Name]},
 year={2024}
}
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments
• Dataset provided by Kaggle Blood Cells Dataset
• CoatNet architecture inspired by Google Research
• Implementation built using TensorFlow/Keras framework
Contact
For questions or collaborations, please contact:
• Author: Ahmad Shaf
• Email: ahmadshaf@cuisahiwal.edu.pk
• Institution: CUI, Sahiwal Campus
