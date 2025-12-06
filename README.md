# Bird Species Classification Using Lightweight CNN & Attention-Enhanced Audio Feature Processing

## Overview
This project implements a lightweight CNN model for bird species classification based on audio recordings. The model uses attention mechanisms like Channel and Spatial Attention and efficient preprocessing techniques to classify bird species with low computational resources.

The project addresses two main challenges:
1. **Computational Burden**: Reducing resources needed for training and deployment.
2. **Limited Generalization**: Improving model robustness across diverse bird sound datasets.

The model is trained using the **Freefield1010** and **Warblr10k** datasets and is optimized for real-world ecological applications.

## Installation
1. Clone the repository:
    ```bash
    git clone <repo_url>
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Preprocess Audio**: 
    ```bash
    python preprocessing.py
    ```
2. **Apply Data Augmentation**: 
    ```bash
    python augmentations.py
    ```
3. **Train the Model**:
    ```bash
    python train.py
    ```

## Datasets
- **Freefield1010**: [Download from Kaggle](https://www.kaggle.com/datasets/raghaw/freefield1010)
- **Warblr10k**: [Download from Kaggle](https://www.kaggle.com/datasets/birdclef2023)

## Model Architecture
The model uses **EfficientNet B0** with added **Channel Attention** and **Spatial Attention** mechanisms to extract features from Mel spectrograms. This improves classification while reducing computational costs.

## Data Augmentation
Augmentation techniques like **Frequency Masking**, **Time Masking**, and **Additive Noise** are used to improve generalization.

## Results
The model achieved **88.67% accuracy** on the test set. Evaluation metrics include accuracy, precision, recall, and F1-score.

## References
1. Duan, L., Yang, L., & Guo, Y. (2024). SIAlex: Species Identification and Monitoring Based on Bird Sound Features. *Ecological Informatics*, 81, 102637. [DOI](https://doi.org/10.1016/j.ecoinf.2024.102637)
2. Xie, J., & Zhu, C. (2023). Acoustic Classification of Bird Species Using Early Fusion of Deep Features. *Applied Sciences*, 13(5), 2931. [DOI](https://doi.org/10.3390/app13052931)

## License
MIT License

