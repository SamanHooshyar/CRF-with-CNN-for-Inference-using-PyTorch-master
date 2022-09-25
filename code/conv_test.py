import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import Conv
import torchtestcase


class ConvTest(torchtestcase.TorchTestCase):
	def test(self):
		#output from custom implementation
		X = torch.Tensor([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]])
		X = X.reshape((1,1,5,5)) # (batch_size x channel_size x height x weight)
		conv = Conv()
		K = torch.Tensor([[[[1,0,1],[0,1,0],[1,0,1]]]])
		conv.init_params(K,3,stride=1,padding=1)
		output_custom = conv.forward(X)
		print(output_custom)

		#ouput from pytorch's implementation
		output_pytorch = F.conv2d(X, K, padding=1,stride=1)
		print(output_pytorch)
		self.assertEqual(output_custom,output_pytorch)
		
		
convTest = ConvTest()
convTest.test()



