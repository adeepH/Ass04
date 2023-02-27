# Assignment 04

Contains codes and files to reproduce results in Assignment 04
Applying Machine Learning in Computational Linguistics

## Installation

1) Clone the repository
2) run ``` pip install -r requirements.txt ```
3) To generate the train, test, dev feature vector files run 

` python get_features.py -df '/path/to/WaseemDatset.txt' -lex '/path/to/lexicon.txt`

## Results
### Wiegand Lexicon
```
Baseline Dev Accuracy:70.12% (Default Parameters)
Baseline Accuracy: 69.92% (Default Parameters)

Modified Parameters: 
-a (Algorithm) IGTREE (Information Gain Tree)
-o (metric) Overlap (IGTREE test runs only Overlap metric)
-w (Weight parameters) IGTREE uses Information Gain weighting

New Accuracy: 70.41%
```
### Lexicon Small
    Same parameter as Wiegand Lexicon
    Baseline Test Accuracy: 67.2%
    New Accuracy: 67.3%

### Inference:
    Most algorithms gave similar results, and the testing accuracy was close to each other. Simliar to how all BERT models score very close to each other.
## License

[Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)