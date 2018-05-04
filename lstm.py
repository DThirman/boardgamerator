from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import torch
from torch.nn.parameter import Parameter
import random
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim


torch.manual_seed(1)

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

EMBEDDING_DIM = 6
HIDDEN_DIM = 64


class LSTMTagger(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, vocab_size):
	    super(LSTMTagger, self).__init__()
	    self.hidden_dim = hidden_dim

#	    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

	    # The LSTM takes word embeddings as inputs, and outputs hidden states
	    # with dimensionality hidden_dim.
	    self.lstm = nn.LSTM(vocab_size, hidden_dim)

	    # The linear layer that maps from hidden state space to tag space
	    self.hidden2out = nn.Linear(hidden_dim, vocab_size)
	    self.hidden = self.init_hidden()
	    self.dropout = nn.Dropout(0.1)

	def init_hidden(self):
	    # Before we've done anything, we dont have any hidden state.
	    # Refer to the Pytorch documentation to see exactly
	    # why they have this dimensionality.
	    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
	    return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
	            autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

	def forward(self, sentence):
	    embeds = sentence #self.word_embeddings(sentence)
	    lstm_out, self.hidden = self.lstm(
	        embeds.view(len(sentence), 1, -1), self.hidden)
	    out = self.hidden2out(lstm_out.view(len(sentence), -1))
	    tag_scores = F.log_softmax(out, dim=1)
	    output = self.dropout(tag_scores)
	    return output

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, n_letters)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0005)

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

def RandomTrainingExample(line):
 	
 	input_line_tensor = Variable(inputTensor(line))
 	target_line_tensor = Variable(targetTensor(line))
 	return input_line_tensor, target_line_tensor

max_length = 64
# Sample from a category and starting letter
def sample(start_letter='A'):
    input = Variable(inputTensor(start_letter))
    hidden = model.init_hidden()

    output_name = start_letter

    for i in range(max_length):
        output = model(input[0])
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
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

import random
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    count = 0
    total = 0
    random.shuffle(lines)
    for line in lines:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        # sentence_in = prepare_sequence(sentence, word_to_ix)
        # targets = prepare_sequence(tags, tag_to_ix)
        input_line_tensor, target_line_tensor = RandomTrainingExample(line)
        loss = 0
        count += 1
        for i in range(input_line_tensor.size()[0]):
        	output = model(input_line_tensor[i])
        	temp = loss_function(output, target_line_tensor[i])
        	#print("temp", temp.data[0])
        	loss += temp
        total += loss

        if count % 100 == 0:
            print(float(count)*100.0/len(lines), "%: ", loss.data[0])

        # # Step 3. Run our forward pass.
        # tag_scores = model(sentence_in)

        # # Step 4. Compute the loss, gradients, and update the parameters by
        # #  calling optimizer.step()
        
        loss.backward()
        optimizer.step()
    print("average loss: ", total/count)
    samples('abcdefghiklmnopqrstuvwxyz')

# See what the scores are after training
# inputs = prepare_sequence(training_data[0][0], word_to_ix)
# tag_scores = model(inputs)
# # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
# #  for word i. The predicted tag is the maximum scoring tag.
# # Here, we can see the predicted sequence below is 0 1 2 0 1
# # since 0 is index of the maximum value of row 1,
# # 1 is the index of maximum value of row 2, etc.
# # Which is DET NOUN VERB DET NOUN, the correct sequence!
# print(tag_scores)