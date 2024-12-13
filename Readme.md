# Food Object Detection and Classification

This repository contains implementations of two distinct approaches to tackle food object detection and classification as part of the **Bangkit 2024 Capstone Project.** The goal is to explore and develop efficient solutions for identifying and categorizing food items in images, using cutting-edge deep learning techniques.

---

### Project Structure

1. Normal CNN for Food Classification

    The normal_cnn directory contains the implementation of a standard Convolutional Neural Network (CNN) for image classification. This was an experimental approach aimed at understanding the fundamentals of food classification before diving into more complex detection models. It served as a preliminary step to:

   - Understand image classification using custom CNN architectures.
   - Experiment with dataset preprocessing, augmentation, and training strategies.

   - Build foundational knowledge for the subsequent Mask R-CNN implementation.

2. Mask R-CNN for Food Detection

    The mrcnn directory houses the implementation of Mask R-CNN, a state-of-the-art model for object detection and instance segmentation. This approach is the core of the project and focuses on detecting and segmenting multiple food items within an image.

    Mask R-CNN was chosen for its ability to localize and classify multiple objects, offering both bounding box detection and pixel-level segmentation. This makes it a powerful tool for applications such as food analysis and portion size estimation.

---
### Why Two Approaches?

 During the early phases of the project, the normal_cnn approach was implemented to experiment with classification techniques. The initial assumption was that Mask R-CNN could adapt and integrate a custom CNN architecture for classification tasks. However, upon deeper exploration, it became clear that Mask R-CNN operates differently, leveraging pre-trained backbones for feature extraction.

 This realization led to a pivot in focus towards implementing Mask R-CNN for detection and segmentation while retaining the normal_cnn implementation as a valuable learning experience. The knowledge gained from both approaches enriched the project, offering insights into:

 - Image classification fundamentals through CNNs.

 - Object detection and segmentation using advanced architectures like Mask R-CNN.

---

### Key Features

**Normal CNN**

- A straightforward convolutional architecture for classifying food images into predefined categories.
- Training pipeline includes data augmentation and validation.
- Insights gained through this implementation provided foundational knowledge for more advanced techniques.

**Mask R-CNN**

- Implementation leverages mrcnn for object detection and instance segmentation.

- Pre-trained backbones like ResNet101 are used for feature extraction.

- Custom configuration for training on food datasets, including:

    - Adjusted anchor scales for food object sizes.

    - Fine-tuning on specific layers for improved performance.

- Extensive use of data augmentation to improve robustness.

# Getting Started

### Conda Environment Setup

Before running any scripts, ensure you have set up the Conda environment using the provided `environment.yaml` file in the root directory. Run the following command to create the environment:

```bash
conda env create -f environment.yaml
```

Activate the environment using:
```bash
conda activate your_env_name
```

### Normal CNN

1. Navigate to the normal_cnn directory.

2. For training, run the Jupyter Notebook food_classification.ipynb.

3. For inference, execute the following script:
```bash
python food_inference.py
```

### Mask R-CNN

1. Navigate to the `mrcnn` directory.

2. Ensure you have installed the requirements from requirements.txt within the Conda environment.
```bash
pip install -r requirements.txt
```

3. The mrcnn directory mainly is organized into three key subdirectories:

   - training: Contains scripts for training the Mask R-CNN model. This includes:

       - A basic Mask R-CNN script for experimental purposes.

       - An advanced Mask R-CNN script for improved performance.

   - utils: Provides utility scripts for dataset preprocessing and randomization.

   - inference: Includes scripts for testing and generating predictions on new images.

For training or inference, simply run the respective Python scripts. Example:
```bash
python training/advanced_mrcnn.py

python inference/test_inference.py
```

# Deployment

The final Mask R-CNN model has been successfully deployed using Flask for scalability. Additionally, a local GUI has been developed for testing purposes. This GUI allows users to interact with the model, upload images, and visualize predictions in real-time. Both the Flask application and the GUI provide seamless access to the model’s detection and segmentation capabilities.
