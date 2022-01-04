# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from torch.utils.data import Dataset, DataLoader

def average_word_embedding(sentence, word_embeddings):
    words = [word_embeddings.get_embedding(word) for word in sentence]
    return torch.from_numpy(np.average(words, axis=0)).float()

class ReviewsDataset(Dataset):
    def __init__(self, train_exs, word_embeddings):
        self.data = []
        for ex in train_exs:
            self.data.append((average_word_embedding(ex.words, word_embeddings), torch.tensor([ex.label]).float()))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NN(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        n_hidden = 50
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, model, word_embeddings):
        self.model = model
        self.word_embeddings = word_embeddings

    def predict(self, ex_words):
        word_embedding = average_word_embedding(ex_words, self.word_embeddings)[None]
        return int(self.model(word_embedding) >= 0)


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    dataset = ReviewsDataset(train_exs, word_embeddings)
    data = DataLoader(dataset, num_workers=0, batch_size=32, shuffle=True, drop_last=False)
    model = NN(word_embeddings.get_embedding_length()).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss = torch.nn.BCEWithLogitsLoss()
    for _ in range(10):
        for exs, labels in data:
            o = model(exs)
            loss_val = loss(o, labels)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
    model.eval()
    return NeuralSentimentClassifier(model, word_embeddings)
