# BREAST-CANCER-PREDICTION

# Breast Cancer Detection Project

## Table of Contents
1. [Project Overview] (#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Methodology](#methodology)
   - [Data Preprocessing] (#data-preprocessing)
   - [Model Selection] (#model-selection)
   - [Model Training and Evaluation] (#model-training-and-evaluation)
6. [Results](#results)
7. [Future Work] (#future-work)
8. [Contributing](#contributing)
9. [License](#license)

## Project Overview
This project aims to detect breast cancer using machine learning techniques. We use a dataset containing various features extracted from breast mass images to classify tumors as either malignant or benign.

## Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Data Set. It includes features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image.

-Source: [UCI Machine Learning Repository] (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download)
- Features: 30 numeric, predictive attributes and the class
- Target variable: Diagnosis (M = malignant, B = benign)


## Installation
To run this project, you need Python 3.7+ and the following libraries:

## Usage
1.	Clone this repository:

2.	Navigate to the project directory:

3.	3. Run the Google Colab notebook:


## Methodology

### Data Preprocessing
1. Load and explore the dataset
2. Handle missing values (if any)
3. Encode categorical variables
4. Split the data into training and testing sets
5. Scale the features

### Model Selection
We chose two models for this project:

1. Logistic Regression:
- Simple and interpretable
- Works well for linearly separable classes
- Provides probability scores
- Computationally efficient

2. Random Forest:
- Handles non-linear relationships well
- Robust to outliers and noise
- Provides feature importance
- Reduces overfitting through ensemble learning

These models were selected for their complementary strengths:
- Logistic Regression offers simplicity and interpretability, which is valuable in medical applications.
- Random Forest captures complex patterns and provides insights into feature importance.

By comparing these models, we can balance between simplicity and complexity, and between linear and non-linear approaches.

### Model Training and Evaluation
1. Train both Logistic Regression and Random Forest models
2. Use cross-validation to assess model performance
3. Optimize Logistic Regression using Randomized Search
4. Evaluate final models on the test set
5. Compare models based on accuracy, precision, recall, and F1-score

## Results
Our breast cancer detection model achieved excellent performance on the test set. Here are the detailed results:

### Classification Report
precision    recall f1-score   support

   False     1.00      0.98      0.99        43
    True     0.99      1.00      0.99        71

accuracy                         0.99       114

macro avg 0.99 0.99 0.99 114 
weighted avg 0.99 0.99 0.99 114

### Confusion Matrix
[[42 1]
 [ 0 71]]

### Interpretation
- The model achieved an overall accuracy of 99%.
- For benign tumors (False):
  - Precision: 100%
  - Recall: 98%
  - F1-score: 99%
- For malignant tumors (True):
  - Precision: 99%
  - Recall: 100%
  - F1-score: 99%
- The confusion matrix shows that out of 114 test cases:
  - 42 benign tumors were correctly identified
  - 71 malignant tumors were correctly identified
  - Only 1 benign tumor was misclassified as malignant
  - No malignant tumors were misclassified as benign

### Single Observation Prediction

For a sample observation, the model predicted:
Prediction: Malignant
 Probability of Malignant: 0.8046375135710787 
Probability of Benign: 0.1953624864289213
This demonstrates the model's ability to not only classify but also provide probability estimates for each class.

### Conclusion

The high accuracy and balanced performance across both classes indicate that our model is highly effective at distinguishing between benign and malignant breast tumors. The model shows particularly strong performance in identifying malignant tumors, which is crucial for early detection and treatment of breast cancer.

However, it's important to note that while the model's performance is impressive, it should be used as a supportive tool in conjunction with expert medical opinion and not as a standalone diagnostic method.

## Future Work
- Experiment with other algorithms (e.g., SVM, Neural Networks)
- Perform more extensive feature engineering
- Collect more data to improve model robustness
- Deploy the model as a web application for easy use by healthcare professionals

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
