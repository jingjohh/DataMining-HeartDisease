
# Heart Disease Data Mining Framework

Data Mining Project - IT160IU, Fall 2026

## 1. Project Overview

This project aims to develop a Data Mining Framework
for Heart Disease Classification using Java and Weka.

The project focuses on:
- Data preprocessing and exploratory data analysis
- Classification using multiple machine learning algorithms
- Model improvement
- Performance evaluation and comparison

## 2. Technologies

- Java (JDK 17+)
- Apache Maven
- Weka API 3.8.7
- Git & GitHub

Python may also be used for EDA and visualization.

## 3. Classification Algorithms

The following algorithms are planned:

1. J48 Decision Tree
2. Random Forest
3. Naive Bayes
4. OneR
5. SimpleLogistic

## 4. Project Structure

    data/
      raw/              # Original dataset
      processed/        # Processed datasets

    src/
      main/java/heartdisease/
        core/           # Shared components
        preprocessing/  # Step 1
        classification/ # Step 2
        improvement/    # Step 3
        evaluation/     # Step 4
        Main.java

      test/java/heartdisease/

    models/             # Saved models
    results/            # Experiment outputs

    pom.xml
    README.md
    .gitignore

## 5. Setup Instructions

### Clone Repository

    git clone https://github.com/jingjohn/DataMining-HeartDisease.git

    cd DataMining-HeartDisease

### Check Java and Maven

    java -version
    javac -version
    mvn -version

JDK 17 or newer and Maven are required.
The project is configured to compile for Java 17.

### Install Dependencies and Build

    mvn clean compile

Maven will automatically download Weka dependencies.

### Run the Demo

    mvn org.codehaus.mojo:exec-maven-plugin:3.5.1:java -Dexec.mainClass=heartdisease.Main

Note: Main.java currently contains a small sample dataset
for testing Weka integration.

The complete training and evaluation pipeline will be
implemented during development.

## 6. Dataset

Dataset: Heart Disease Dataset

Source:
https://www.kaggle.com/datasets/oktayrdeki/heart-disease

Download the dataset and place the CSV file at:

    data/raw/heart_disease.csv

The dataset is excluded from Git.
Each member must download it separately.

## 7. Team Responsibilities

| Step | Responsibility |
|------|----------------|
| Step 1 | Data Preprocessing & EDA |
| Step 2 | Classification Algorithms |
| Step 3 | Model Improvement |
| Step 4 | Model Evaluation & Integration |

All members contribute to the final report.

## 8. Git Workflow

Each member works on their assigned feature branch.

Available branches:

- feature/preprocessing
- feature/classification-a
- feature/classification-b
- feature/improvement
- feature/evaluation

Workflow:

1. Switch to your assigned branch.
2. Pull the latest updates.
3. Implement and test your changes.
4. Commit and push to your feature branch.
5. Create a Pull Request to main.
6. Wait for review before merging.

Do not push directly to main.

## 9. Evaluation

Planned evaluation includes:

- Stratified 10-fold Cross-Validation
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Training and Prediction Runtime

Preprocessing must avoid data leakage during evaluation.

## 10. Project Status

Project initialization completed.

Current focus:
- Dataset preprocessing
- Classification implementation
- Model improvement design
- Evaluation framework development
