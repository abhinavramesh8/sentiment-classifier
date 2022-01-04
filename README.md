# Sentiment Classifier

## Requirements

Python 3.5+ and PyTorch 1.0+  
Using CPU alone is enough.

## Data and Task

We will be using the movie review dataset of Socher et al. (2013). This is a dataset of movie review snippets taken from Rotten Tomatoes. The task we are dealing with is this: positive/negative binary sentiment classification of sentences, with neutral sentences discarded from the dataset.

The data files consist of newline-separated sentiment examples, with a label (0 or 1) followed by a tab, followed by the sentence, which has been tokenized but not lowercased. The data has been split into a train, development (dev), and blind test set. The blind test set only has the sentences without the labels.

We accomplish this task using a deep averaging network as discussed in Iyyer et al. (2015). If we have an input consisting of several words, we average the real-valued vector embeddings of all those words, which is then fed as input to a feedforward neural network that does the prediction.

There are two sources of pretrained embeddings that can be used: `data/glove.6B.50d-relativized.txt` and `data/glove.6B.300d-relativized.txt`. These are trained using GloVe (Pennington et al., 2014). These vectors have been relativized to the data.

## Usage

Clone this repository and `cd` into it. Then, run the following command:

`python3 neural_sentiment_classifier.py --word_vecs_path <embeddings file path>`

For example,

`python3 neural_sentiment_classifier.py --word_vecs_path data/glove.6B.300d-relativized.txt`

This will load in the data, initialize the feature extractor, train the model, evaluate it on train, dev and blind test sets, write the blind test set results to a file and output the dev and train set accuracy, precision and recall as well as the the total time for training and evaluation.