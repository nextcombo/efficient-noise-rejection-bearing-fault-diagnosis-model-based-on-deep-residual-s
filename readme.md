# Efficient Noise Rejection! Bearing Fault Diagnosis Model Based on Deep Residual Shrinkage Network (DRSN)

Preface
This issue is based on the bearing dataset from Case Western Reserve University (CWRU), combined with the Deep Residual Shrinkage Networks (DRSN) for research on bearing fault classification. By introducing residual shrinkage modules, this effectively suppresses noise and irrelevant information, enhancing the model's sensitivity to key features. A 1D convolutional DRSN model based on PyTorch framework has been implemented, evaluating its classification performance on different faults of the CWRU dataset. The experimental results show that the proposed model outperforms ordinary residual networks in terms of accuracy and robustness, verifying the effectiveness of the deep residual shrinkage mechanism in bearing fault diagnosis.
Dataset: Bearing Dataset from Case Western Reserve University (CWRU)
Environmental Framework: Python 3.11, PyTorch 2.1 and above are required to run
Accuracy: 100% on the test set
Usage Objective: Beginners, paper requirements, graduation design requirements
Code Guarantee: Code comments are detailed, ready to run without modification.
Introduction to Deep Residual Shrinkage Networks (DRSN)
1.1 Model Overview
When faced with high-noise vibration signals, the learning ability of ResNet can decline. Because ResNet uses convolutional kernels as local feature extractors, due to the interference of noise, it might not be able to detect relevant fault features, resulting in poor discrimination power at the output layer, which is insufficient for accurate fault classification. Therefore, a new deep learning method is necessary for fault diagnosis based on vibration signals of rotating motors under strong background noise.
DRSN consists of residual networks with a shrinkage module added. The shrinkage module acts similar to an attention mechanism, controlling the activation degree of feature channels through a gating structure, automatically "soft thresholding" the signal, suppressing invalid information. This structure is particularly suitable for mechanical vibration signals containing noise and interference, enhancing the representation capability of key fault features.
(1) DRSN-CS
DRSN-CS is a variant of the ResNet network that uses soft thresholding to eliminate noise-related features. Soft thresholding is inserted as a non-linear transformation layer within structural units. Simultaneously, this threshold can be learned within these structural units. The unit is named (RSBU-CS, i.e., Residual shrinkage building unit). Unlike RBU (Residual building unit, the basic unit in ResNet), RSBU-CS has a special module for estimating the threshold. In this module, GAP is used to obtain a one-dimensional vector from the absolute values of feature maps x. Subsequently, this one-dimensional vector is passed through two fully connected layers to obtain scale parameters. After the two fully connected layers, a sigmoid function is used, at which point the scale parameters are converted into a value between 0 and 1. This scale parameter is then multiplied by the average of |x| to obtain the threshold. This step is considered because the threshold needs to be positive and not too large, otherwise, after soft thresholding, the features will all become zero. The network structure of RSBU-CS and DRSN-CS is shown in the following figure:
\t (2)\tDRSN-CW  
DRSN-CW is another variant of ResNet, where DRSN-CS applies a threshold to all channels, while DRSN-CW sets a threshold for each channel. The feature map x is dimensionally reduced into a one-dimensional vector and then input into two fully connected layers. The number of neurons in the second fully connected layer corresponds to the number of input feature maps' channels, after which each neuron undergoes sigmoid activation. Finally, the scale values of specific channels are multiplied by the average value of the specific channels in the input feature map, resulting in the thresholds for each channel. The structure is shown below:
1.2 Model Overall Design
Based on the 1D-ResNet convolutional neural network architecture, a DRSN model is designed by combining the deep residual contraction module. The model workflow includes:
Input Layer: Single-channel time series signals with input shape (batch_size, channel=1, seq_len=1024).
Initial Convolutional Layer: Extracting basic temporal sequence features.
Multiple Residual Shrinkage Blocks: Each block consists of two sub-convolutional layers and a shrinkage module to strengthen effective signals.
Pooling and Classification Layer: Through adaptive pooling, reduce dimensions, and finally output the probability of fault categories using fully connected layers.
Deep Residual Shrinkage Module: The residual block includes two 1D convolutions, batch normalization, and ReLU activation followed by an introduction of the shrinkage module (Shrinkage Module):
The shrinkage module generates channel attention weights through global average pooling on feature channels.
This module processes the input features "soft thresholding", suppressing noise and irrelevant features, enhancing the response to key features.
The addition of this module enables the network to dynamically adjust channel weights, improving robustness.
Data Preprocessing for CWRU Data
2.1 Import Data
Refer to previous articles for preprocessing of fault classification in 10 categories, using the Case Western Reserve University bearing data \t10-category dataset:
2.2 Data Preprocessing
Create datasets by sliding windows and uniformly divide them into training sets, validation sets, and test sets at a ratio of 7:2:1, and finally save the data.
Fault Diagnosis Model Based on DRSN
3.1 Define the network model, set parameters, and train the model
50 epochs, accuracy 100%, significantly improves classification results with deep residual shrinkage networks (DRSN) - the loss rapidly decreases during the initial training period, model accuracy gradually increases, the validation set curve is stable, indicating good convergence of the network without obvious overfitting, and can extract important features from fault signal characteristics that are crucial for model recognition, with significant effects!
3.2 Model Evaluation
1. Accuracy, Precision, Recall, F1 Score
2. Failure Classification Confusion Matrix:
3. Update Download
1. 27 Bearing Fault Diagnosis Models | The most cost-effective family set across the network
27 Bearing Fault Diagnosis Models | The most cost-effective family set across the network
Download Link:
Previously Interesting Content:
Python - Kaiser Permanente University (CWRU) Bearing Data Interpretation and Classification Processing
Python Bearing Fault Diagnosis (Part 1): Short-Time Fourier Transform STFT
Python Bearing Fault Diagnosis (Part 2): Continuous Wavelet Transform CWT
Python Bearing Fault Diagnosis (Part 3): Empirical Mode Decomposition EMD
Pytorch-LSTM Bearing Fault One-Dimensional Signal Classification (Part 1)
Pytorch-CNN Bearing Fault One-Dimensional Signal Classification (Part 2)
Pytorch-Transformer Bearing Fault One-Dimensional Signal Classification (Part 3)
Python Bearing Fault Diagnosis (Part 4): Fault Classification Based on EMD-CNN
Python Bearing Fault Diagnosis (Part 5): Fault Classification Based on EMD-LSTM
Python Bearing Fault Diagnosis (Part 6): Fault Classification Based on EMD-Transformer
Python Bearing Fault Diagnosis (Part 7): Fault Classification Based on EMD-CNN-LSTM
Python Bearing Fault Diagnosis (Part 8): Fault Classification Based on EMD-CNN-GRU Parallel Model
Bearing Fault Identification Model based on FFT + CNN - BiGRU-Attention Attention Fusion of Time and Frequency Features
Bearing Fault Identification Model based on FFT + CNN - Transformer Attention Fusion of Time and Frequency Features
Python Bearing Fault Diagnosis (Part 9): Fault Classification Based on VMD+CNN-BiLSTM
Python Bearing Fault Diagnosis (10) - VMD+CNN-Transformer for Fault Classification
Python Bearing Fault Diagnosis (11) - VMD+CNN-BiGRU-Attention for Fault Classification
Cross Attention Fusion of Time and Frequency Features with FFT + CNN-BiLSTM-CrossAttention Bearing Fault Identification Model
Cross Attention Fusion of Time and Frequency Features with FFT + CNN-Transformer-CrossAttention Bearing Fault Identification Model
Introduction to Python Bearing Fault Diagnosis
Python Bearing Fault Diagnosis (13) - Super-strong Machine Learning Recognition Model Based on Fault Signal Feature Extraction
Python Bearing Fault Diagnosis (14) - Highly Innovative Fault Recognition Model
Python Bearing Fault Diagnosis (15) - One-Dimensional Fault Signal Recognition Model Based on CNN-Transformer
Python Bearing Fault Diagnosis (16) - Highly Innovative Fault Recognition Model (II)
SCI 1 Region High Innovation Bearing Fault Diagnosis Model!
Python Bearing Fault Diagnosis (18) - Innovative Diagnosis Model Based on CNN-TCN-Attention
Based on GADF+Swin-CNN-GAM High Innovation Bearing Fault Diagnosis Model
Attention Magic Modification | Super-strong Bearing Fault Diagnosis Model!
Bearing Fault—Cross Attention Innovative Model Full Set
Python Bearing Fault Diagnosis (19) - Innovative Diagnosis Model Based on Transformer-BiLSTM
END

## Images

![High-efficiency noise resistance! Bearing fault diagnosis model based on deep residual shrinkage network (DRSN)](11.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
