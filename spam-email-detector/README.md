# 📧 Spam Email Detector

## Overview
This project detects whether a message is **Spam** or **Ham (Not Spam)** using Machine Learning.

## Technologies Used
- Python
- Pandas
- Scikit-learn

## Machine Learning Algorithm
- Multinomial Naive Bayes

## Features
- Uses the real SMS Spam Collection Dataset
- Converts text into numerical features using CountVectorizer
- Trains a Naive Bayes classifier
- Displays model accuracy
- Displays a classification report
- Predicts whether a custom message is spam or ham

## Dataset
- SMS Spam Collection Dataset
- 5,574 labeled SMS messages

## Files

main.py - Main application

SMSSpamCollection - Dataset

## How to Run

```bash
python main.py
```

## Example

Input:

```
Congratulations! You won a FREE prize!
```

Output:

```
Prediction: spam
```