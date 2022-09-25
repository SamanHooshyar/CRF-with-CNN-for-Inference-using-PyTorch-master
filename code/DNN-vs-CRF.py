# -*- coding: utf-8 -*-

import gzip
import csv
import numpy as np
import torchvision.models as models
from PIL import Image
import torch

# memory footprint support libraries/code
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
!pip install gputil
!pip install psutil
!pip install humanize
import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]
def printm():
 process = psutil.Process(os.getpid())
 print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
 print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()



t_2d = torch.randn(2,4)
print(t_2d)
torch.max(t_2d,1)

class DataLoader:
    def __init__(self):
        data_path = 'letter.data.gz'
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



#Define Network Architecture
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
#     super(Net, self).__init__()
#     self.input_size=input_size
#     self.conv1 = nn.Conv2d(1, 3, 3,padding=1)
#     self.conv2 = nn.Conv2d(3, 10, 3)
#     self.fc1 = nn.Linear(96, hidden_size1)
#     self.fc2 = nn.Linear(hidden_size2, output_size)
#     self.dropout = nn.Dropout(0.2)
    
    super(Net,self).__init__()
    self.input_size=input_size
    self.conv1 = nn.Conv2d(1, 3, 3,padding=1)
    self.conv2 = nn.Conv2d(3, 16, 3,padding=1)
    self.conv3 = nn.Conv2d(16, 64, 3,padding=1)
    #self.conv4 = nn.Conv2d(64, 96, 3,padding=1)
    self.fc1 = nn.Linear(int(64*input_size/4), hidden_size1)
    self.fc2 = nn.Linear(hidden_size1, hidden_size2)
    self.fc3 = nn.Linear(hidden_size2,output_size)
    self.dropout = nn.Dropout(0.5) # dropout prevents overfitting of data
    self.conv1_bn = nn.BatchNorm2d(3) #batch normalization
    self.conv2_bn = nn.BatchNorm2d(16) #batch normalization
    self.conv3_bn = nn.BatchNorm2d(64) #batch normalization
    
  def forward(self,x):
#     x = x.view(-1,self.input_size).view(-1,16,8).unsqueeze(1)
#     out = F.relu(self.conv1(x))
#     #print(x.shape)
#     out = F.max_pool2d(out, (2,2))           
#     #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#     #print(x.shape)
#     #Before feeding to fully-connected layer, features should be flattened
#     out = out.view(-1, self.num_flat_features(out))
#     #print(x.shape)
#     out = F.relu(self.fc1(out))
#     out = self.dropout(out)
#     out = self.fc2(out)
#     out = self.dropout(out)
#     return out
    
    #flatten image input
    #x = x.view(-1,self.input_size).view(-1,16,8).unsqueeze(1)
    out = F.relu(self.conv1(x))
    #out = F.max_pool2d(out, (2,2))
    out = self.conv1_bn(out)   
    out = F.relu(self.conv2(out))
    out = self.conv2_bn(out)
    out = F.relu(self.conv3(out))
    out = self.conv3_bn(out)
    out = F.max_pool2d(out, (2,2))
    
    #out = F.relu(self.conv1(out))
    #add hidden layer, with relu activation function
    out = out.view(-1, int((self.input_size*64)/4))
      
    out = self.fc1(out)
    out = F.relu(out)
    out = self.dropout(out)
    out = self.fc2(out)
    out = F.relu(out)
    out = self.dropout(out)
    out = self.fc3(out)
    
    #upto this point produces class scores
    #out = F.log_softmax(out,dim=1) #changes class scores to class probabilities
    return out
  
  def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

import torch
import torch.optim as optim
import torch.utils.data as data_utils
#from data_loader import get_dataset
import numpy as np


# Tunable parameters
batch_size = 256
num_epochs = 100
max_iters  = 2000
print_iter = 100 # Prints results every n iterations
conv_shapes = [[1,64,128]] #


# Model parameters
input_dim = 128
embed_dim = 64
num_labels = 26
cuda = torch.cuda.is_available()

# Instantiate the CRF model
#crf = CRF(input_dim, embed_dim, conv_shapes, num_labels, batch_size)

#input_size = 128
hidden_size1 = 512 #Do research for these hyperparameters
hidden_size2 = 512
#output_size = 26
#initialize the NN




#Custome Model
model = Net(input_dim,hidden_size1, hidden_size2,num_labels)

#RESNET18
#model = models.resnet18(pretrained=True)
#model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#ALEXNET
#model = models.alexnet()

#SQUEEZENET1_0
#model = models.squeezenet1_0(pretrained=True)

#VGG16
#model = models.vgg16(pretrained=True)

#DENSENET161
#model = models.densenet161(pretrained=True)

#INCEPTION_V3
#model = models.inception_v3(pretrained=True)

#GOOGLENET
#model = models.googlenet(pretrained=True)

# Setup the optimizer
#opt = optim.LBFGS(model.parameters())
opt = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

##################################################
# Begin training
##################################################
step = 0

# Fetch dataset
dataset = get_dataset()

for i in range(num_epochs):
    print("Processing epoch {}".format(i))
    dataset = get_dataset()
    split = int(0.5 * len(dataset.data)) # train-test split
    train_data, test_data = dataset.data[:split], dataset.data[split:]
    train_target, test_target = dataset.target[:split], dataset.target[split:]

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

    # Now start training
    for i_batch, sample in enumerate(train_loader):
        train_X = sample[0]
        train_Y = sample[1]
        
        if cuda:
            train_X = train_X.cuda()
            train_Y = train_Y.cuda()
            model.cuda()
        
        
        train_Y = train_Y.view(-1,num_labels)
        # compute loss, grads, updates:
        
        #train_X = train_X.view(-1,input_dim)
        #train_X = train_X.unsqueeze(1).unsqueeze(1)
        
        train_X = train_X.view(-1,128).view(-1,16,8).unsqueeze(1)
        
        def closure():
          opt.zero_grad() # clear the gradients
          output = model(train_X)
          tr_loss = criterion(output, torch.max(train_Y, 1)[1]) # Obtain the loss for the optimizer to minimize
#           if step % print_iter == 0:
#             print("training loss: ")
#             print(step, tr_loss.data,
#                        tr_loss.data / batch_size)
          
          tr_loss.backward() # Run backward pass and accumulate gradients
          return tr_loss
        
       
        tr_loss = opt.step(closure) # Perform optimization step (weight updates)

        # print to stdout occasionally:
        if step % print_iter == 0:
            random_ixs = np.random.choice(test_data.shape[0], batch_size, replace=False)
            test_X = test_data[random_ixs, :]
            test_Y = test_target[random_ixs, :]
          
            # Convert to torch
            test_X = torch.from_numpy(test_X).float()
            test_Y = torch.from_numpy(test_Y).long()
            
            
            
            
            if cuda:
                test_X = test_X.cuda()
                test_Y = test_Y.cuda()
            
            test_X = test_X.view(-1,128).view(-1,16,8).unsqueeze(1)
            test_Y = test_Y.view(-1,num_labels)
            
            sum_vec = test_Y.sum(dim=1)
            #print(list(test_Y))
            test_output = model(test_X)
            #print(list(test_output))
            test_loss = criterion(test_output, torch.max(test_Y, 1)[1])
            #print("step,training loss,test loss,average training loss, average test loss")
            print(step, tr_loss.data.item(), test_loss.data.item(),
                       tr_loss.data.item() / batch_size, test_loss.data.item() / batch_size)

			##################################################################
			# IMPLEMENT WORD-WISE AND LETTER-WISE ACCURACY HERE
			##################################################################

            #print(blah)
            prediction_indices = torch.max(test_output,1)[1]
            target_indices = torch.max(test_Y,1)[1]
            
            calculate_accuracy(prediction_indices,target_indices)
            
            
        
        

        step += 1
        if step > max_iters: raise StopIteration
    del train, test

def calculate_accuracy(prediction_indices,target_indices):

  correct = 0            
  zero_padded_count = 0
  for i in range(0,len(prediction_indices)):
    if prediction_indices[i]==target_indices[i] and sum_vec[i]!=0:
       correct+=1
    if sum_vec[i]==0:
       zero_padded_count +=1
  accuracy = correct/ (test_Y.shape[0]-zero_padded_count)
  print("accuracy: " + str(accuracy))
            
    
            
  #Create a view to batch_size X max_word_length, so that word-wise comparison is possible
  prediction_indices_v = prediction_indices.view(batch_size,-1)
  target_indices_v = target_indices.view(batch_size,-1)

  num_correct_word = 0
  #loop over instances in batch and calculate if all letters in particular instance are predicted correctly
  for i in range(0,prediction_indices_v.shape[0]):
    num_correct_word+= torch.all(torch.eq(prediction_indices_v[i],target_indices_v[i])).item()
  
  accuracy_word_wise = num_correct_word/batch_size
  print(accuracy_word_wise)

