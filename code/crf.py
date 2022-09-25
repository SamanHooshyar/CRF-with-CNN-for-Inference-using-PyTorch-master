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

def get_crf_obj(train_X, train_Y, W, T, C):
    print("in get_crf_obj")
    log_likelihood = torch.tensor([0.0], requires_grad=True)
    #W_t = torch.transpose(W)
    W_t = W
    # n = len(word_list)
    #Log-likelihood calculation
    for i, X in enumerate(train_X):
        Y = train_Y[i]
        z = calculate_log_z(X, len(Y), W_t, T)
        z_x = z
        node_poten = torch.tensor([0.0], requires_grad=True)
        edge_poten = torch.tensor([0.0], requires_grad=True)
        for s in range(len(Y)):
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
        for s in range(len(Y)-1):
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
        self.W = nn.Parameter(torch.randn(26, 128, requires_grad=True))
        self.T = nn.Parameter(torch.randn(26, 26, requires_grad=True))
        # self.allParameters=nn.Parameter(torch.randn(26*128+26*26))

        ### Use GPU if available
        # if self.use_cuda:
        #     [m.cuda() for m in self.modules()]

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
        # convfeatures = blah
        # return convfeatures
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
