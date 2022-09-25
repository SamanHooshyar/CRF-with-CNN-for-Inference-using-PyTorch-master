import torch
import torch.nn as nn
from math import log
from string import ascii_lowercase
import numpy as np

def forward(X, m, W, T):
    # T = np.transpose(T)
    alpha = np.zeros((m, 26))
    for i in range(1, m):
        for j in range(26):
            total_sum = []
            for k in range(26):
                 total_sum.append(np.dot(W[k], X[i-1]) + T[k,j] + alpha[i-1, k])
            temp = log_sum_exp(np.array(total_sum))
            alpha[i, j] = temp
    return alpha

def backward(X, m, W, T):
    beta = np.zeros((m, 26))
    # T = np.transpose(T)
    for i in range(m-2, -1, -1):
        for j in range(26):
            total_sum = []
            for k in range(26):
                total_sum.append(np.dot(W[k], X[i+1]) + T[j,k] + beta[i+1, k])
            temp = log_sum_exp(np.array(total_sum))
            beta[i, j] = temp
    return beta

def log_sum_exp(arr):
    M = arr.max()
    return log(np.sum(np.exp(np.add(arr, -1*M)))) + M

def calculate_log_z(X, m, W, T):
    alpha = forward(X, m, W, T)
    z = []
    for i in range(26):
        z.append(np.add(np.dot(W[i], X[m-1]), alpha[m-1, i]))
    return log_sum_exp(np.array(z))

def gradient_w(word_list, W, T, C):
    print("in gradient_w")
    grad_w = np.zeros((26, 128))
    indicator = np.zeros(26)
    #W_t = np.transpose(W)
    W_t = W
    count = 0
    #print(word_list)
    for X, Y in word_list:
        count += 1
        #print("current count:")
        #print(count)
        m = len(Y)
        alpha = forward(X, m, W_t, T)
        beta = backward(X, m, W_t, T)
        log_z = calculate_log_z(X, m, W_t, T)
        temp_grad = np.zeros((26, 128))
        for s in range(m):
            prob = np.add(alpha[s,:], beta[s,:])
            # node = np.matmul(np.transpose(W), X[s])
            # node = np.matmul(W[Y[s]], X[s])
            node = np.matmul(W_t, X[s])
            prob = np.add(prob, node)
            prob = np.add(prob, -1*log_z)
            prob = np.exp(prob)

            indicator[Y[s]] = 1
            indicator = np.add(indicator, -1*prob)
            letter_grad = np.tile(X[s], (26, 1))
            out = np.multiply(indicator[:, np.newaxis], letter_grad)
            temp_grad = np.add(out, temp_grad)
            indicator[:] = 0
        grad_w = np.add(grad_w, temp_grad)
    grad_w = np.multiply(grad_w, -1*C/count)
    grad_w = np.add(grad_w, W_t)
    return grad_w.flatten()

def gradient_t(word_list, W, T, C):
    print("in gradient_t")
    grad_t = np.zeros((26, 26))
    indicator = np.zeros((26, 26))
    #W_t = np.transpose(W)
    W_t = W
    count = 0
    for X, Y in word_list:
        count += 1
        #print("current count:")
        #print(count)
        m = len(Y)
        alpha = forward(X, m, W_t, T)
        beta = backward(X, m, W_t, T)
        log_z = calculate_log_z(X, m, W_t, T)
        temp_grad = np.zeros((26, 26))
        for s in range(m-1):
            node = np.add.outer(np.matmul(W_t, X[s]), np.matmul(W_t, X[s+1]))
            node = np.add(node, T)
            node = np.add(alpha[s][:, np.newaxis], node)
            node = np.add(beta[s+1], node)
            prob = np.add(-1*log_z, node)
            prob = np.exp(prob)
            indicator[Y[s], Y[s+1]] = 1
            out = np.add(indicator, -1*prob)
            temp_grad = np.add(out, temp_grad)
            indicator[:, :] = 0
        grad_t = np.add(grad_t, temp_grad)
    grad_t = np.multiply(grad_t, -1*C/count)
    grad_t = np.add(grad_t, T)
    return grad_t.flatten()
    # return temp_grad

def get_crf_obj(word_list, W, T, C):
    print("in get_crf_obj")
    #print("W.shape")
    #print(W.shape)
    #print("T.shape")
    #print(T.shape)
    log_likelihood = 0.0
    #W_t = np.transpose(W)
    W_t = W
    n = len(word_list)
    #log-likelihood calculation
    for X,Y in word_list:
        z = calculate_log_z(X, len(Y), W_t, T)
        z_x = z
        node_poten = 0.0
        edge_poten = 0.0
        for s in range(len(Y)):
            y_s = Y[s]
            #print("y_s")
            #print(y_s)
            #print("W_t[y_s]")
            #print(W_t[y_s])
            #print("X[s]")
            #print(X[s])
            node_poten += np.dot(W_t[y_s], X[s])
        for s in range(len(Y)-1):
            edge_poten += T[Y[s]][Y[s+1]]
        
        p_y_x = node_poten + edge_poten - z_x
        #print("pyx")
        #print(p_y_x)
        #print("node_poten")
        #print(node_poten)
        #print("edge_poten")
        #print(edge_poten)
        #print("z_x")
        #print(z_x)
        log_likelihood += p_y_x

    # norm_w calculation
    norm_w = [] 
    for i in range(26):
        norm_w.append(np.linalg.norm(W_t[i]))
    norm_w = np.sum(np.square(norm_w))

    #norm_t calculation
    norm_t = np.sum(np.square(T))
    return -log_likelihood
    #return -1*(C/n)*log_likelihood + 0.5 * norm_w + (0.5 * norm_t)
  
def max_sum(X,W,T):
    word_size = len(X)  # 100
    l = np.zeros((word_size,len(T))) # 100 * 26
    y = np.zeros((word_size)) # 100

    #print("X, W, T, l:")
    #print(X.shape)
    #print(W.shape)
    #print(T.shape)
    #print(l.shape)
    for i in range(1, word_size):  # in max-sum algorithm first we store values for l recursively: O(100 * 26 * 26) = O(|Y|m^2)
        for y_i in range(0,26):
            l[i][y_i] = max([np.dot(W[j], X[i-1]) + T[j][y_i] + l[i-1][j] for j in range(0,26)])

###############  recovery part in max-sum algorithm 

    m = word_size-1 # 99
    max_sum = [np.dot(W[y_m],X[m]) + l[m][y_m] for y_m in range(0,26)]  # O(26)
    y[m] = np.argmax(max_sum)
    max_sum_value = max(max_sum)
    #print("max objective value:", max_sum_value)

    for i in range(m, 0, -1):   # O(m * 26)
        y[i-1] = int(np.argmax([np.dot(W[j],X[i-1]) + T[j][int(y[i])] + l[i-1][j] for j in range(0,26)]))

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
    #word = word.cpu().detach().numpy()
    Y = []
    for j in range(len(word)):
        Y.append(np.argmax(word[j]))
    return Y

class CRF(nn.Module):
    def __init__(self, input_dim, embed_dim, conv_layers, num_labels, batch_size):
        #raise
        """
        Linear chain CRF as in Assignment 2
        """
        super(CRF, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.conv_layers = conv_layers
        self.num_labels = num_labels
        self.batch_size = batch_size
        #self.transition=nn.Parameter(torch.randn(26,26))
        self.use_cuda = torch.cuda.is_available()
        self.allParameters=nn.Parameter(torch.randn(26*128+26*26))

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

    def getW(self):
        return self.allParameters.cpu().detach().numpy()[:26*128].reshape(26,128)
        
    def getT(self):
        return self.allParameters.cpu().detach().numpy()[26*128:].reshape(26,26)

    def predict(self, X, W, T):
        print("in predict")
        features = self.get_conv_features(X).cpu().detach().numpy()
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
        return self.predict(X, self.getW(), self.getT())
        #print(X)
        #features = self.get_conv_features(X).cpu().detach().numpy()
        #prediction = []
        #for i in range(len(X)):
        #    Xi = features[i]
        #    prediction.append(max_sum(Xi, self.getW(), self.getT()))
        #return prediction

    def loss(self, X, labels):
        #raise
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """
        print("in loss")
        #print(X.shape)
        X = self.get_conv_features(X).cpu().detach().numpy()
        loss = 0
        Y = labels.cpu().detach().numpy()
        W = self.getW()
        W = W.reshape(26,128)
        T = self.getT()
        T = T.reshape(26,26)
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
        #XY = zip(X.cpu().detach().numpy(),Y.cpu().detach().numpy())
        print("length XY:")
        print(len(XY))
        C = 1
        print("computing gradients")
        self.grad_w = gradient_w(XY, W, T, C)
        self.grad_t = gradient_t(XY, W, T, C)
        print("computing loss")
        loss = get_crf_obj(XY, W, T, C)
        #print(loss)
        #TODO: convert to a tensor
        return torch.tensor(loss).float()
        
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
        Y = Y.cpu().detach().numpy()
        total = len(Y)
        correct = 0.00
        for i in range(len(Y)):
            print("predicted.shape")
            print(predicted[i].shape)
            print(predicted[i])
            print("actual, predicted:")
            print(getAsciiWord(getAsciiVal(Y[i])))
            print(getAsciiWord(predicted[i]))
            
            if np.array_equal(getAsciiVal(Y[i]), predicted[i]):
                correct += 1.00
        return correct/total * 100
    
    def computeModelAccuracy(self, X, Y, W, T):
        predicted = self.predict(X,W,T)
        Y = Y.cpu().detach().numpy()
        total = len(Y)
        correct = 0.00
        for i in range(len(Y)):
            print("predicted.shape")
            print(predicted[i].shape)
            print(predicted[i])
            print("actual, predicted:")
            print(getAsciiWord(getAsciiVal(Y[i])))
            print(getAsciiWord(predicted[i]))
            if np.array_equal(getAsciiVal(Y[i]), predicted[i]):
                correct += 1.00
        return correct/total * 100
