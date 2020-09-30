import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cpu")

class LSTMClassifier(nn.Module):

	def __init__(self, input_size, num_layers, hidden_size, seq_len, num_classes):

		super(LSTMClassifier, self).__init__()

		self.hidden_size	=	hidden_size
		self.num_layers 	= 	num_layers
		self.lstm			=	nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		#self.fc				=	nn.Linear(hidden_size*seq_len, num_classes)
		self.fc1			=	nn.Linear(hidden_size*seq_len, self.hidden_size)
		self.fc2			=	nn.Linear(self.hidden_size, num_classes)
		#self.drop			=	nn.Dropout(p=0.2)


	def forward(self, x):

		h0		=	torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		c0 		=	torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		out, _ 	= 	self.lstm(x, (h0,c0))
		# out: tensor of shape (batch_size, seq_length, hidden_size)
		output 	= 	out.reshape(out.shape[0], -1)
		#output 	= 	self.fc(output)
		output	=	self.fc1(output)
		output	=	self.fc2(output)
		return output
