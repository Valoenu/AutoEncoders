# Import Libraries
import pandas as pd  # For data handling
import numpy as np  # For numerical arrays

# Torch Libraries
import torch
import torch.nn as nn  # To build neural network layers
import torch.optim as optim  # For optimization algorithms
from torch.autograd import Variable  # To convert tensors into variables for training

# Importing the dataset
users_dataset = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
films_dataset = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings_dataset = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Training and Test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
number_of_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
number_of_films = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Function to convert the dataset into a user-movie matrix
def convert(data):
    new_data = []
    for user_id in range(1, number_of_users + 1):
        film_ids = data[:, 1][data[:, 0] == user_id]
        ratings = data[:, 2][data[:, 0] == user_id]
        user_ratings = np.zeros(number_of_films)
        user_ratings[film_ids - 1] = ratings
        new_data.append(list(user_ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convert to Torch Tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Building the architecture of the Autoencoder
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(number_of_films, 20)  # Encoder layer 1
        self.fc2 = nn.Linear(20, 10)               # Encoder layer 2
        self.fc3 = nn.Linear(10, 20)               # Decoder layer 1
        self.fc4 = nn.Linear(20, number_of_films)  # Decoder output
        self.activation = nn.Sigmoid()             # Activation function

    def forward(self, x):
        x = self.activation(self.fc1(x))  # First encoding
        x = self.activation(self.fc2(x))  # Second encoding
        x = self.activation(self.fc3(x))  # First decoding
        x = self.fc4(x)                   # Output layer (no activation)
        return x

# Initialize the Autoencoder
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# Training the SAE
epochs_number = 180
for epoch in range(1, epochs_number + 1):
    train_loss = 0
    s = 0.
    for user_id in range(number_of_users):
        input = Variable(training_set[user_id]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.requires_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = number_of_films / float(torch.sum(target.data > 0)) + 1e-10
            loss.backward()
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1.
            optimizer.step()
            optimizer.zero_grad()
    print(f'Epoch: {epoch} | Training Loss: {train_loss / s:.4f}')

# Testing the SAE
test_loss = 0
s = 0.
for user_id in range(number_of_users):
    input = Variable(training_set[user_id]).unsqueeze(0)
    target = Variable(test_set[user_id])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.requires_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = number_of_films / float(torch.sum(target.data > 0)) + 1e-10
        test_loss += np.sqrt(loss.item() * mean_corrector)
        s += 1.
print(f'\n✅ Test Loss: {test_loss / s:.4f}')