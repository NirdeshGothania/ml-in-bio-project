# Automated ML-Based Classification Tool

## Overview
This repository contains an automated machine learning (ML) tool for solving classification problems. The tool follows an ideal ML pipeline, allowing users to preprocess data, apply normalization techniques, perform feature selection, choose cross-validation methods, select ML models, and evaluate the model's performance. The output is presented in a PDF report.

## Project Structure
The project is organized into several Python scripts, each responsible for a specific part of the ML pipeline. The main script orchestrates the entire process.

- `data_loader.py`: Handles data loading, exploration, cleaning, and splitting.
- `data_preprocessing.py`: Implements various normalization/standardization techniques.
- `feature_selection.py`: Provides options for feature selection using different techniques.
- `cross_validation.py`: Includes scripts for different cross-validation techniques.
- `classification_models.py`: Contains scripts for various classification models.
- `report_generator2.py`: Generates a structured report in PDF format.
- `main.py`: Orchestrates the entire ML pipeline based on user inputs.
- `requirements.txt`: Contains all the necessary Python packages needed for the execution.

## Execution
First, make sure that the current directory opened in the terminal is the correct directory in which all the python script files are present alongwith dataset file in .csv format.

1. And make sure you have the required dependencies installed before running the code. You can install them by executing the following command in your terminal:

```
pip install -r requirements.txt
```

This command installs the necessary Python packages listed in the requirements.txt file, including:
- pandas
- scikit-learn
- reportlab
- matplotlib
- mlxtend

Additionally, ensure that you have the necessary dependencies for the MLxtend library, which is used for Sequential Feature Selection. Install it separately using:  

```
pip install mlxtend
```

2. Now change the dataset file name to dataset.csv.
	eg. breast_cancer.csv --> dataset.csv

3. How to run the code; Use this command on the terminal:  
```
python main.py
```

4. **Input Format**: 

You have to type the proper text(method name given in the option) for choosing the option. To configure the data preprocessing and model selection for your machine learning pipeline, follow the following steps:

	i) Choose the normalization/standardization method by specifying one of the options: Standard, MinMax, or Robust.
	ii) Indicate the feature selection method to use by selecting from the options: SelectKBest, RFE, VarianceThreshold, PCA, or SFS.
	iii) Define the cross-validation approach by selecting from the options: kfold, stratified_kfold, or leave_one_out.
	iv) Specify the classification model to employ by choosing from the options: RandomForest, SVM, LogisticRegression, GradientBoosting, or NaiveBayes.

5. **Output Information**:

To view the results of the machine learning pipeline, refer to the following:

	i) An output file named "report.pdf" will be created in the same directory as the main source file main.py.
	ii) The file contains plots on scores of cross-validation and overall metrics on the blind dataset.
