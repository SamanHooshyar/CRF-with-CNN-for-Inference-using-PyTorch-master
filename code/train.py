import torch
import torch.optim as optim
import torch.utils.data as data_utils
from data_loader import get_dataset
import numpy as np
from crf import CRF


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

# Instantiate the CRF model
crf = CRF(input_dim, embed_dim, conv_shapes, num_labels, batch_size)

# Setup the optimizer
# opt = optim.LBFGS(crf.parameters())
opt = optim.Adam(crf.parameters())


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
        # if cuda:
        #     print("cuda : yes")
        #     train_X = train_X.cuda()
        #     train_Y = train_Y.cuda()

        # compute loss, grads, updates:
        print("computing outputs")
        tr_loss = torch.tensor([0.0], requires_grad=True)
        def closure():
            print("in closure")
            outputs = crf(train_X)
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

