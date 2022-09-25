import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    """
    Convolution layer.
    """
    def __init__(self):
        super(Conv, self).__init__()


    def init_params(self,kernel,kernel_size=5,stride=1,padding=0):
        """
        Initialize the layer parameters
        :return:
        """
        
		
        #if kernel is None return default random kernel
        if kernel is None:
          self.kernel = nn.Parameter(torch.rand(kernel_size,kernel_size, requires_grad=True).cuda())
        else:
          self.kernel = kernel
          
        self.kernel_size=kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self,X):
        """
        Forward pass
        :return:
        """
		
        X = F.pad(input=X, pad=(self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        regions_all = X.unfold(2,self.kernel_size,self.stride).unfold(3,self.kernel_size,self.stride)
        fil_batch=[]
        for regions_batch in regions_all:
          fil_channel = []
          for regions_channel in regions_batch:
            fil_row =[]
            for regions_row in regions_channel:
              fil_col = []
              for regions_column in regions_row:
                cur_elem = torch.sum(torch.mul(regions_column,self.kernel))
                fil_col.append(cur_elem.item())
              fil_row.append(fil_col)
            fil_channel.append(fil_row)
        fil_batch.append(fil_channel)

        filtered = torch.FloatTensor(fil_batch)
        return filtered
        	

    def backward(self):
        """
        Backward pass
        :return:
        """




