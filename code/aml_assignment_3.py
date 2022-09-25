# -*- coding: utf-8 -*-
"""Copy of Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AmrXvxZ5zv7sVP-G_xLIP5Gs9bKhy4fG
"""

import gzip
import csv
import numpy as np
import torch
import torch.tensor

class DataLoader:
    def __init__(self):
        data_path = '../data/letter.data.gz'
        lines = self._read(data_path)
        data, target = self._parse(lines)
        self.data, self.target = self._pad(data, target)

    @staticmethod
    def _read(filepath):
        with gzip.open(filepath, 'rt') as file_:
            reader = csv.reader(file_, delimiter='\t')
            lines = list(reader)
            return lines

    @staticmethod
    def _parse(lines):
        lines = sorted(lines, key=lambda x: int(x[0]))
        data, target = [], []
        next_ = None

        for line in lines:
            if not next_:
                data.append([])
                target.append([])
            else:
                assert next_ == int(line[0])
            next_ = int(line[2]) if int(line[2]) > -1 else None
            pixels = np.array([int(x) for x in line[6:134]])
            pixels = pixels.reshape((16, 8))
            data[-1].append(pixels)
            target[-1].append(line[1])
        return data, target

    @staticmethod
    def _pad(data, target):
        """
        Add padding to ensure word length is consistent
        """
        max_length = max(len(x) for x in target)
        padding = np.zeros((16, 8))
        data = [x + ([padding] * (max_length - len(x))) for x in data]
        target = [x + ([''] * (max_length - len(x))) for x in target]
        return np.array(data), np.array(target)

def get_dataset():
    dataset = DataLoader()

    # Flatten images into vectors.
    dataset.data = dataset.data.reshape(dataset.data.shape[:2] + (-1,))

     # One-hot encode targets.
    target = np.zeros(dataset.target.shape + (26,))
    for index, letter in np.ndenumerate(dataset.target):
        if letter:
            target[index][ord(letter) - ord('a')] = 1
    dataset.target = target

    # Shuffle order of examples.
    order = np.random.permutation(len(dataset.data))
    dataset.data = dataset.data[order]
    dataset.target = dataset.target[order]
    return dataset

import torch
import torch.nn as nn
from math import log
from string import ascii_lowercase
# import numpy as np

def forward(X, m, W, T):
    # T = torch.transpose(T)
    alpha = torch.zeros((m, 26))
    for i in range(1, m):
        for j in range(26):
            total_sum = []
            for k in range(26):
                 total_sum.append(torch.dot(W[k], X[i-1]) + T[k,j] + alpha[i-1, k])
            temp = log_sum_exp(torch.tensor(total_sum))
            alpha[i, j] = temp
    return alpha

def backward(X, m, W, T):
    beta = torch.zeros((m, 26))
    # T = torch.transpose(T)
    for i in range(m-2, -1, -1):
        for j in range(26):
            total_sum = []
            for k in range(26):
                total_sum.append(torch.dot(W[k], X[i+1]) + T[j,k] + beta[i+1, k])
            temp = log_sum_exp(torch.tensor(total_sum))
            beta[i, j] = temp
    return beta

def log_sum_exp(arr):
    M = arr.max()
    return log(torch.sum(torch.exp(torch.add(arr, -1*M)))) + M

def calculate_log_z(X, m, W, T):
    alpha = forward(X, m, W, T)
    z = []
    for i in range(26):
        z.append(torch.add(torch.dot(W[i], X[m-1]), alpha[m-1, i]))
    return log_sum_exp(torch.tensor(z))

def gradient_w(train_X, train_Y, W, T, C):
    print("in gradient_w")
    grad_w = torch.zeros(26, 128, requires_grad=True)
    # indicator = torch.zeros(26, requires_grad=True)
    indicator = torch.zeros(26)
    #W_t = torch.transpose(W)
    W_t = W
    count = 0
    #print(word_list)
    for i, X in enumerate(train_X):
        Y = train_Y[i]
        count += 1
        #print("current count:")
        #print(count)
        m = len(Y)
        alpha = forward(X, m, W_t, T)
        beta = backward(X, m, W_t, T)
        log_z = calculate_log_z(X, m, W_t, T)
        temp_grad = torch.zeros((26, 128))
        for s in range(m):
            prob = torch.add(alpha[s,:], beta[s,:])
            # node = torch.matmul(torch.transpose(W), X[s])
            # node = torch.matmul(W[Y[s]], X[s])
            node = torch.matmul(W_t, X[s])
            prob = torch.add(prob, node)
            prob = torch.add(prob, -1*log_z)
            prob = torch.exp(prob)

            indicator[Y[s]] = torch.ones(1)
            indicator = torch.add(indicator, -1*prob)
            # letter_grad = torch.tile(X[s], (26, 1))
            Xs = X[s]
            letter_grad = Xs.repeat(26, 1)
            out = torch.multiply(indicator[:, torch.newaxis], letter_grad)
            temp_grad = torch.add(out, temp_grad)
            indicator[:] = 0
        grad_w = torch.add(grad_w, temp_grad)
    grad_w = torch.multiply(grad_w, -1*C/count)
    grad_w = torch.add(grad_w, W_t)
    return grad_w

def gradient_t(train_X, train_Y, W, T, C):
    print("in gradient_t")
    grad_t = torch.zeros((26, 26))
    indicator = torch.zeros((26, 26))
    #W_t = torch.transpose(W)
    W_t = W
    count = 0
    for i, X in enumerate(train_X):
        count += 1
        Y = train_Y[i]
        #print("current count:")
        #print(count)
        m = len(Y)
        alpha = forward(X, m, W_t, T)
        beta = backward(X, m, W_t, T)
        log_z = calculate_log_z(X, m, W_t, T)
        temp_grad = torch.zeros((26, 26))
        for s in range(m-1):
            node = torch.add.outer(torch.matmul(W_t, X[s]), torch.matmul(W_t, X[s+1]))
            node = torch.add(node, T)
            node = torch.add(alpha[s][:, torch.newaxis], node)
            node = torch.add(beta[s+1], node)
            prob = torch.add(-1*log_z, node)
            prob = torch.exp(prob)
            indicator[Y[s], Y[s+1]] = 1
            out = torch.add(indicator, -1*prob)
            temp_grad = torch.add(out, temp_grad)
            indicator[:, :] = 0
        grad_t = torch.add(grad_t, temp_grad)
    grad_t = torch.multiply(grad_t, -1*C/count)
    grad_t = torch.add(grad_t, T)
    return grad_t.flatten()
    # return temp_grad

def getWordLength(wordImage):
    for i, letterImage in enumerate(wordImage):
        if torch.max(torch.eq(torch.zeros(16, 8).cuda(), letterImage)) == 0:
            return i
    
def get_crf_obj(train_X, train_Y, W, T, C):
    print("in get_crf_obj")
    log_likelihood = torch.tensor([0.0], requires_grad=True).cuda()
    #W_t = torch.transpose(W)
    W_t = W
    # n = len(word_list)
    #Log-likelihood calculation
    for i, X in enumerate(train_X):
        Y = train_Y[i]
        lenWord = getWordLength(X)
        z = calculate_log_z(X, lenWord, W_t, T)
        # z = calculate_log_z(X, len(Y), W_t, T)
        z_x = z
        node_poten = torch.tensor([0.0], requires_grad=True).cuda()
        edge_poten = torch.tensor([0.0], requires_grad=True).cuda()
        # for s in range(len(Y)):
        for s in range(lenWord):
            y_s = torch.argmax(Y[s])
            # print(y_s)
            # print(y_s.shape)
            Wys = W_t[y_s.item()]
            # print("W_t shape:")
            # print(W_t.shape)
            # print("Wys shape:")
            # print(Wys.shape)
            tmp = node_poten + torch.dot(Wys.view(-1), X[s].view(-1))
            node_poten = tmp
        # for s in range(len(Y)-1):
        for s in range(lenWord):
            ys = torch.argmax(Y[s])
            ys1 = torch.argmax(Y[s+1])
            tmp = edge_poten + T[ys][ys1]
            edge_poten = tmp
        
        p_y_x = node_poten + edge_poten - z_x
        tmp = log_likelihood + p_y_x
        log_likelihood = tmp

    # # norm_w calculation
    # norm_w = [] 
    # for i in range(26):
    #     norm_w.append(torch.linalg.norm(W_t[i]))
    # norm_w = torch.sum(torch.square(norm_w))

    # #norm_t calculation
    # norm_t = torch.sum(torch.square(T))
    return -log_likelihood
    #return -1*(C/n)*log_likelihood + 0.5 * norm_w + (0.5 * norm_t)
  
def max_sum(X,W,T):
    word_size = len(X)  # 100
    l = torch.zeros((word_size,len(T))) # 100 * 26
    y = torch.zeros((word_size)) # 100

    for i in range(1, word_size):  # in max-sum algorithm first we store values for l recursively: O(100 * 26 * 26) = O(|Y|m^2)
        for y_i in range(0,26):
            l[i][y_i] = max([torch.dot(W[j], X[i-1]) + T[j][y_i] + l[i-1][j] for j in range(0,26)])

###############  recovery part in max-sum algorithm 

    m = word_size-1 # 99
    max_sum = torch.tensor([torch.dot(W[y_m],X[m]) + l[m][y_m] for y_m in range(0,26)], requires_grad=True)  # O(26)
    y[m] = torch.argmax(max_sum)
    max_sum_value = max(max_sum)
    #print("max objective value:", max_sum_value)

    for i in range(m, 0, -1):   # O(m * 26)
        y[i-1] = int(torch.argmax(torch.tensor([torch.dot(W[j],X[i-1]) + T[j][int(y[i])] + l[i-1][j] for j in range(0,26)], requires_grad=True)))

    return y

mapping = list(enumerate(ascii_lowercase))
alphaToVal = { i[1]:i[0] for i in mapping }
valToAlpha = { i[0]:i[1] for i in mapping }
  
def getAsciiWord(word):
    Y = []
    for j in range(len(word)):
        #print(j)
        #print(word[j])
        #print(mapping[word[j]])
        Y.append(valToAlpha[word[j]])
    return Y

def getAsciiVal(word):
    #word = word
    Y = []
    for j in range(len(word)):
        Y.append(torch.argmax(word[j]))
    return Y

class CRF(nn.Module):
    def __init__(self, input_dim, embed_dim, conv_layers, num_labels, batch_size):
        #raise
        """
        Linear chain CRF as in Assignment 2
        """
        print("in CRF::__init__")
        super(CRF, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.conv_layers = conv_layers
        self.num_labels = num_labels
        self.batch_size = batch_size
        #self.transition=nn.Parameter(torch.randn(26,26))
        self.use_cuda = torch.cuda.is_available()
        self.W = nn.Parameter(torch.randn(26, 128, requires_grad=True).cuda())
        self.T = nn.Parameter(torch.randn(26, 26, requires_grad=True).cuda())
        # self.allParameters=nn.Parameter(torch.randn(26*128+26*26))
        
        self.kernel = nn.Parameter(torch.randn(5, 5, requires_grad=True).cuda())

        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]

    def init_params(self):
        raise
        print("init_params")
        """
        Initialize trainable parameters of CRF here
        """
        #blah
        #self.allParameters=nn.Parameter(torch.randn(26*128+26*26))

    # def getW(self):
    #     return self.allParameters[:26*128].reshape(26,128)
        
    # def getT(self):
    #     return self.allParameters[26*128:].reshape(26,26)

    def predict(self, X, W, T):
        print("in predict")
        print(X.size())
        features = self.get_conv_features(X)
        prediction = []
        for i in range(len(X)):
            Xi = features[i]
            prediction.append(max_sum(Xi, W, T))
        return prediction
    
    def forward(self, X):
        """
        Implement the objective of CRF here.
        The input (features) to the CRF module should be convolution features.
        """
        print("in forward")
        print(X.size())
        return self.predict(X, self.W, self.T)
        #print(X)
        #features = self.get_conv_features(X)
        #prediction = []
        #for i in range(len(X)):
        #    Xi = features[i]
        #    prediction.append(max_sum(Xi, self.W, self.T))
        #return prediction

    def loss(self, X, labels):
        #raise
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """
        print("in loss")
        print(X.size())
        #print(X.shape)
        X = self.get_conv_features(X)
        loss = 0
        Y = labels
        W = self.W
        # W = W.reshape(26,128)
        T = self.T
        # T = T.reshape(26,26)
        XY = []
        #mapping = list(enumerate(ascii_lowercase))
        #mapping = { i[1]:i[0] for i in mapping }
        for i in range(len(Y)):
            word = Y[i]
            #print("word")
            #print(word)
            Y_ = getAsciiVal(word)
            #print("Y_")
            #print(Y_)
            XY.append((X[i],Y_))
        #XY = zip(X,Y)
        print("length XY:")
        print(len(XY))
        C = 1
        print("computing gradients")
        # self.grad_w = gradient_w(X, Y, W, T, C)
        # self.grad_t = gradient_t(X, Y, W, T, C)
        print("computing loss")
        loss = get_crf_obj(X, Y, W, T, C)
        return loss
        #print(loss)
        #TODO: convert to a tensor
        # return torch.tensor(loss).float()
        
    # def backward(self):
    #     """
    #     Return the gradient of the CRF layer
    #     :return:
    #     """
    #     print("in backward")
    #     #gradient = torch.zeros(26*128+26*26)
    #     #print(self.grad_w)
    #     #print(self.grad_w.shape)
    #     #print(self.grad_t.shape)
    #     #print(self.grad_t)
    #     return torch.from_numpy(np.concatenate((self.grad_w, self.grad_t)))

    def get_conv_features(self, X):
        """
        Generate convolution features for a given word
        """
        
        fil_row =[]
        for regions_row in regions_channel:
            fil_col = []
            for regions_column in regions_row:
                cur_elem = torch.sum(torch.mul(regions_column,self.kernel))
                fil_col.append(cur_elem.item())
                fil_row.append(fil_col)
        return X

    def wordAccuracy(self, X, Y):
        predicted = self.forward(X)
        Y = Y
        total = len(Y)
        correct = 0.00
        for i in range(len(Y)):
            print("predicted.shape")
            print(predicted[i].shape)
            print(predicted[i])
            print("actual, predicted:")
            print(getAsciiWord(getAsciiVal(Y[i])))
            print(getAsciiWord(predicted[i]))
            
            if torch.eq(getAsciiVal(Y[i]), predicted[i]):
                correct += 1.00
        return correct/total * 100
    
    def computeModelAccuracy(self, X, Y, W, T):
        predicted = self.predict(X,W,T)
        Y = Y
        total = len(Y)
        correct = 0.00
        for i in range(len(Y)):
            print("predicted.shape")
            print(predicted[i].shape)
            print(predicted[i])
            print("actual, predicted:")
            print(getAsciiWord(getAsciiVal(Y[i])))
            print(getAsciiWord(predicted[i]))
            if torch.eq(getAsciiVal(Y[i]), predicted[i]):
                correct += 1.00
        return correct/total * 100

import torch
import torch.optim as optim
import torch.utils.data as data_utils
#from data_loader import get_dataset
import numpy as np
#from crf import CRF


# Tunable parameters
batch_size = 256
num_epochs = 100
max_iters  = 1000
print_iter = 25 # Prints results every n iterations
conv_shapes = [[1,64,128]] #


# Model parameters
input_dim = 128
embed_dim = 64
num_labels = 26
cuda = torch.cuda.is_available()
print("cuda : ")
print(cuda)

# Instantiate the CRF model
crf = CRF(input_dim, embed_dim, conv_shapes, num_labels, batch_size)

# Setup the optimizer
# opt = optim.LBFGS(crf.parameters())
opt = optim.Adam(crf.parameters())

from conv import Conv

convLayer = Conv()
convLayer.init_params(kernel_size=5)

##################################################
# Begin training
##################################################
step = 0

# Fetch dataset
dataset = get_dataset()
# split = int(0.5 * len(dataset.data)) # train-test split
split = int(0.01 * len(dataset.data)) # train-test split
# train_data, test_data = dataset.data[:split], dataset.data[split:]
# train_target, test_target = dataset.target[:split], dataset.target[split:]
train_data = dataset.data[:split]
test_data = train_data
train_target = dataset.target[:split]
test_target = train_target

# Convert dataset into torch tensors
train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())

# Define train and test loaders
train_loader = data_utils.DataLoader(train,  # dataset to load from
                                     batch_size=batch_size,  # examples per batch (default: 1)
                                     shuffle=True,
                                     sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                     num_workers=5,  # subprocesses to use for sampling
                                     pin_memory=False,  # whether to return an item pinned to GPU
                                     )

test_loader = data_utils.DataLoader(test,  # dataset to load from
                                    batch_size=batch_size,  # examples per batch (default: 1)
                                    shuffle=False,
                                    sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                    num_workers=5,  # subprocesses to use for sampling
                                    pin_memory=False,  # whether to return an item pinned to GPU
                                    )
print('Loaded dataset... ')

for epoch in range(num_epochs):
    print("Processing epoch {}".format(epoch))

    batch_size = 68
    running_loss = 0.0

    # Now start training
    # print("train_loader.size")
    # print(train_loader.size())
    for i, data in enumerate(train_loader):
        train_X, train_Y = data
        print("i, train_X, train_Y")
        print(i)
        print(train_X.shape)
        print(train_Y.shape)
        if cuda:
            print("cuda : yes")
            train_X = train_X.cuda()
            train_Y = train_Y.cuda()

        # compute loss, grads, updates:
        print("computing outputs")
        tr_loss = torch.tensor([0.0], requires_grad=True).cuda()
        def closure():
            print("in closure")
            tmp = convLayer(train_X)
            outputs = crf(tmp)
            opt.zero_grad() # clear the gradients
            print("gradients cleared")
            print("computing loss")
            tr_loss = crf.loss(train_X, train_Y) # Obtain the loss for the optimizer to minimize
            print("starting backprop")
            tr_loss.backward() # Run backward pass and accumulate gradients
            print("completed backprop. tr_loss : %d" % (tr_loss.item()) )
            return tr_loss.item()

        opt.step(closure) # Perform optimization step (weight updates)

        # print statistics
        print('epoch, loss : %d, %f' % (epoch, tr_loss.item()))
        
	##################################################################
	# IMPLEMENT WORD-WISE AND LETTER-WISE ACCURACY HERE
	##################################################################
        step += 1
        if step > max_iters: raise StopIteration
    # del train, test

print("finished training")
print(crf.W)
print(crf.T)