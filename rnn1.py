from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import torch
import random
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

def randomTrainingLine():
    line = randomChoice(lines)
    return line

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

def RandomTrainingExample():
 	line = randomTrainingLine()
 	input_line_tensor = Variable(inputTensor(line))
 	target_line_tensor = Variable(targetTensor(line))
 	return input_line_tensor, target_line_tensor


f=open('games.txt')
prev = 0
allChars = set("")
lines = []
for line in f.readlines():
	line = line.encode('ascii', 'ignore')
	lines.append(line)
	for c in line:
		allChars.add(c)
chars = ""
for c in allChars:
	chars += c
all_letters = ''.join(sorted(chars))
n_letters = len(all_letters) + 1
rnn = RNN(n_letters, 256, n_letters)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.0005)
learning_rate = 0.0005

def train(input_line_tensor, target_line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(input_line_tensor[i], hidden)
        # print(output);l'p
        # print(target_line_tensor[i])
        loss += loss_function(output, target_line_tensor[i])

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0] / input_line_tensor.size()[0]

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

n_iters = 100000
print_every = 50
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()


for iter in range(1, n_iters + 1):
    output, loss = train(*RandomTrainingExample())
    #print(loss)
    total_loss += loss
    #print(total_loss)
    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)


max_length = 70

# Sample from a category and starting letter
def sample(start_letter='A'):
    input = Variable(inputTensor(start_letter))
    hidden = rnn.initHidden()

    output_name = start_letter

    for i in range(max_length):
        output, hidden = rnn(input[0], hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0][0]
        if topi == n_letters - 1:
            break
        else:
            letter = all_letters[topi]
            output_name += letter
        input = Variable(inputTensor(letter))

    return output_name

# Get multiple samples from one category and multiple starting letters
def samples(start_letters='ABC'):
    for i in range(100):
    	start_letter = random.choice(start_letters)
        print(sample(start_letter))



samples('abcdefghiklmnopqrstuvwxyz')
	



