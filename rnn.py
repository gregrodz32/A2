import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import string
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser 

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        _, hidden = self.rnn(inputs)
        # Compute representations after RNN
        representations = hidden.squeeze(0)  # Squeeze to remove batch dimension
        # Output layer computation
        output = self.W(representations)
        predicted_vector = self.softmax(output)
        return predicted_vector


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)  # Fill in parameters
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open('/Users/greg/Library/Mobile Documents/com~apple~CloudDocs/UTD/Spring 2024/SE 4375/assignment2_release/Data_Embedding/word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    while not stopping_condition:
        random.shuffle(train_data)
        model.train()

        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]

                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)

                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()

        train_loss = loss_total / loss_count
        print("Training Loss for epoch {}: {}".format(epoch + 1, train_loss))
        train_losses.append(train_loss)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)

        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data

        val_loss_total = 0
        val_loss_count = 0

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1

            val_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))
            val_loss_total += val_loss
            val_loss_count += 1

        val_loss_avg = val_loss_total / val_loss_count
        val_losses.append(val_loss_avg.item())

        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        validation_accuracy = correct / total
        val_accuracies.append(validation_accuracy)

        if validation_accuracy < last_validation_accuracy and train_accuracy > last_train_accuracy:
            stopping_condition=True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = train_accuracy

        epoch += 1

    # Plotting the accuracies
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
