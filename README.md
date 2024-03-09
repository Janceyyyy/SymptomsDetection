# Reddit Mental Health Analysis Project

## Overview
This project involves analyzing discussions from Reddit to explore mental health-related topics. Utilizing advanced natural language processing (NLP) techniques, it aims to identify and classify discussions around mental health issues, employing a variety of Python libraries and machine learning models.

## Technologies Used
- **Python**: The primary programming language.
- **Google Colab**: Used for running the Python script with cloud computation, enabling easy sharing and collaboration.
- **Pandas & Numpy**: For data manipulation and numerical operations.
- **Scikit-learn**: Employed for machine learning tasks, including the Random Forest Classifier for classification, Cross Validation for model evaluation, CountVectorizer for text feature extraction, and Latent Dirichlet Allocation (LDA) for topic modeling.
- **Happiestfuntokenizing**: A tokenizing library for text processing.
- **Transformers**: Utilizes the RobertaModel and RobertaTokenizer for state-of-the-art NLP capabilities, including text embedding and tokenization.
- **Joblib**: For lightweight pipelining and saving models.

## Project Structure

### Preprocessing
- **Load and Prepare Data**: Functions are provided to load datasets, presumably from pickled files, and prepare them for analysis. This involves cleaning text data and tokenizing using `happiestfuntokenizing` and `RobertaTokenizer`.

### Analysis
- **Feature Extraction and Topic Modeling**: Implements `CountVectorizer` and `Latent Dirichlet Allocation (LDA)` for extracting text features and identifying topics within the mental health discussions.
- **Classification**: Utilizes a Random Forest Classifier to classify discussions, supported by cross-validation techniques to evaluate model performance.

### Model Saving and Loading
- **Joblib**: Used for saving and loading models, enabling the reuse of trained models without retraining.

