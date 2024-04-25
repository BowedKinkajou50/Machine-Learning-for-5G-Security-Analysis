# Machine Learning for 5G Security Analysis
## Idaho National Labratory

## Project Team
- Nicholas Kaminski  - Idaho National Labratory - Mentor
- Milos Manic - Computer Science - Faculty Advisor
- Even Pham - Computer Science - Student Team Member
- James McGlone - Computer Science - Student Team Member
- Miles Fagan - Computer Science - Student Team Member
- Megan Sorochin - Computer Science - Student Team Member

## Overview
This senior capstone project focuses on utilizing machine learning algorithms for analyzing security aspects in 5G networks. It contains implementations of three different machine learning algorithms: decision tree classifier, Gaussian naive bayes classifier, and multi-layer perceptron classifier. These algorithms are applied to datasets within the src folder to analyze and classify security-related data in 5G networks.
## Background
Wireless networks are becoming both increasingly critical to modern life and increasingly complex and difficult to secure.  Applying ML to the analysis of cyber attack graphs, a common way to represent the steps for completing a cyber attack, represents a promising path for supporting the security of modern wireless networks.

## Structure
| Folder | Description |
|---|---|
| Documentation |  all documentation the project team has created to describe the architecture, design, installation and configuratin of the peoject |
| Notes and Research | Relavent information useful to understand the tools and techniques used in the project |
| Status Reports | Project management documentation - weekly reports, milestones, etc. |
| src | Source code - create as many subdirectories as needed |

src/: Contains the source code for the machine learning algorithms and datasets.
- naivebayesddos.py: Implementation of Gaussian naive bayes with the ddos_modified.csv dataset.
- naivebayesfraud.py: Implementation of Gaussian naive bayes with the fraud_modified.csv dataset.
- decisiontree.py: Implementation of decision tree classifier with the ddos_modified.csv dataset.
- decisiontreeFraud.py: Implementation of decision tree classifer with the fraud_modified.csv dataset.
- neuralnetworkddos.py: Implementation of multi-layer perceptron classifier with the ddos_modified.csv dataset.
- neuralnetworksfraud.py: Implementation of multi-layer perceptron classifier with the fraud_modified.csv dataset.
- ddos_modified.csv: The DDoS dataset.
- fraud_modified.csv: The Fraud dataset. 

## Requirements 
To run the code in this project, ensure you have the following dependencies installed:

- Python 3.x
- scikit-learn
- pandas
- numpy

## Usage

- Clone or download this repository to your local machine
- Navigate to the src folder
- Select an algorithm and dataset pair you wish to run
- Results will be displayed in command line
  
