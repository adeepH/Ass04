# Assignment 05

This repository contains the source code for our grad school Assignment 05.

## Overview

Our project aims to solve the identification of sexism and racism in Waseem Dataset. We make use of 

## Installation

To install the project, you need to follow these steps:

1. Clone the repository to your local machine
git clone https://github.com/adeepH/CSCI_B_659-LING-L-645.git 
2. Install the necessary dependencies ```pip install -r requirements.txt```

## Usage

To run the project, you need to follow these steps:

1. Navigate to the project directory, ```cd Assignment-05``` 
2. Run the main script, with the necessary arguments.
3. To see the installed models, run ```python main.py --help```

<br> 
<p align="center">
  <img src="https://github.com/adeepH/CSCI_B_659-LING-L-645/blob/master/Assignment-05/data/help.png" width="60%"/>
<br> 
</p>
<br> 

## Results

1. Ngrams used: 
    - Character ngrams (n_range=(2,4))
    - Word ngrams (n_range=(1,5))
    - We used ```sklearn.feature_extraction.text.CountVectorizer()```
2. ML models used: 
    - Decision Tree Classifier
    - Logistic Regression
    - Multinomial Naive Bayes
    - Random Forest Classifier
    - Adaptive Gardient Boosting (AdaBoost)
    - Support Vector Classifier
3. A snippet of Decision Tree Classifier when used for both Character ngrams and word ngrams:

<br> 
<p align="center">
  <img src="https://github.com/adeepH/CSCI_B_659-LING-L-645/blob/master/Assignment-05/data/res.png" width="60%"/>
<br> 
</p>
<br> 

4. Surprisingly, Decision Tree classifier is scoring 99.96% on the test set with word ngrams and 93.79% on char ngrams, with said ngram range.
5. Other classifiers scored in a similar way, with the scores dropping when ```n>=5 & n <=2```.


## Contributors

- Adeep Hande (ahande@iu.edu)
- Shubham Agarwal (shubagar@iu.edu)
