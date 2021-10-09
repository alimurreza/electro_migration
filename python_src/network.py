import torch
import torch.nn as nn
import pdb

class EMNetv4(nn.Module):
	# network architecture:
	# this network doesn't share the network weights before fusion (see below)
	# input(1xWxH) -> conv features -> fully-connected -> 
	# 							-> fusion -> regression-loss and prediction
	# input(1xWxH) -> conv features -> fully-connected -> 

	def __init__(self, models, fusion_method='concat'):
		super(EMNetv4, self).__init__()

		self.fusion_method = fusion_method
		self.network_A = EMNetv3(models) # for CCDImage
		self.network_B = EMNetv3(models) # for ThermalImage
		
		'''# pdb.set_trace()
		# remove the last layers from the each network
		new_regress_module_A 		= self.update_regress_module(self.network_A)
		self.network_A.regress_module 	= new_regress_module_A

		new_regress_module_B 		= self.update_regress_module(self.network_B)
		self.network_B.regress_module 	= new_regress_module_B'''

		#pdb.set_trace()
		self.regress_module = torch.nn.Sequential()


		# fully-connected layer 4: 128 (nodes) -> 64 (nodes) ( Fusion method torch add, torch multiply)
		#self.regress_module.add_module('fc4', nn.Linear(128, 64))
		#self.regress_module.add_module('relu4', nn.ReLU())
		#self.regress_module.add_module('drop4', nn.Dropout(0.5))


		# fully-connected layer 5: 64 (nodes) -> 32 (nodes)
		self.regress_module.add_module('fc5', nn.Linear(64, 32))
		self.regress_module.add_module('relu5', nn.ReLU())
		self.regress_module.add_module('drop5', nn.Dropout(0.5))


		# fully-connected layer 6: 32 (nodes) -> 1 (node)
		self.regress_module.add_module('fc6', nn.Linear(32, 1))
		# there is no non-linear layer at the output layer.


		# all the fully connected layer will be learned during backprop
		for param in self.regress_module.parameters():
		    param.requires_grad = True
		
	'''# drops the last layer from the fully connected module
	def update_regress_module(self, network):
		mod = list(network.regress_module.children())
		last_fc = mod.pop()		# UNUSED last_fc
		# save the rest
		new_regress_module  = nn.Sequential(*mod) 
		return new_regress_module'''

	def forward(self, x_a, x_b):

		vis = 0
		
		output_a = self.network_A(x_a)
		output_b = self.network_B(x_b)

		
		
		if (vis):
			print("feature map size() after EMnetv3 (input CCD image): {}".format(output_a.size()))
			print("feature map size() after EMnetv3 (input Thermal image): {}".format(output_b.size()))
		# apply the 'fusion_method' and add fully connected layers on the fused features
		if (self.fusion_method == 'concat'):
			output = torch.cat([output_a, output_b], 1)

		elif(self.fusion_method == 'concat_add'):
			output = torch.add(output_a,output_b)
		elif(self.fusion_method == 'concat_mul'):
			output = torch.mul(output_a,output_b)
		elif(self.fusion_method == 'concat_lstm'):
			output = torch.mul(torch.tanh(output_a), torch.sigmoid(output_b))
			
		else:
			print('fusion_method is not defined')
			pdb.set_trace()			

		if (vis):		
			print("feature map size() of fused featurs: {}".format(output.size()))

		output = self.regress_module(output)

		
		if (vis):
			print("feature map size() after regress_module (multiple layers of fcs): {}".format(output.size()))

		return output


# Reza: 07/03/19 - Modified EMNetv2 architecture. added more private variables for fully connected layers of the regress module to access them individually. useful for fusion
class EMNetv3(nn.Module):
	# network architecture:

	# input(1xWxH) -> conv features -> fully-connected 
	# 				   -> regression-loss and prediction

	def __init__(self, models):
		super(EMNetv3, self).__init__()

		model = models.vgg16(pretrained=True)
		model_features = list(model.features.children())

		# first 3 convolutional blocks are used as it is
		conv1_block = model_features[0:5] 	# conv(64)->relu->conv(64)->relu->maxpool
		conv2_block = model_features[5:10] 	# conv(128)->relu->conv(128)->relu->maxpool
		conv3_block = model_features[10:16] # conv(256)->relu->conv(256)->relu->conv(256)->relu

		self.conv1_block = nn.Sequential(*conv1_block)
		self.conv2_block = nn.Sequential(*conv2_block)
		self.conv3_block = nn.Sequential(*conv3_block)

		print(self.conv1_block)
		print(self.conv2_block)
		print(self.conv3_block)

		# initialize parameters and do NOT require gradient computation
		for param in self.conv1_block.parameters():
		    param.requires_grad = False
		for param in self.conv2_block.parameters():
		    param.requires_grad = False
		for param in self.conv3_block.parameters():
		    param.requires_grad = False
		

		maxpool3 = model_features[16] 		# 3rd maxpool layer discarded (REDUNDANT BUT KEPT HERE FOR BETTER UNDERSTANDING)
		# fourth block is modified with dropout layers fused in (see below)
		conv4_block = model_features[17:22]

		# append all the blocks now
		conv4_block_conv1 	= conv4_block[0:2] # conv(512)->relu
		self.conv4_block_conv1 	= nn.Sequential(*conv4_block_conv1)
		self.conv4_block_drop1 	= nn.Dropout(0.5)  # dropout->(NEW)

		conv4_block_conv2 	= conv4_block[2:4] # conv(512)->relu
		self.conv4_block_conv2 	= nn.Sequential(*conv4_block_conv2)
		self.conv4_block_drop2 	= nn.Dropout(0.5)  # dropout->(NEW)

		conv4_block_conv3 	= conv4_block[4:6] # conv(512)->relu
		self.conv4_block_conv3  = nn.Sequential(*conv4_block_conv3)
		self.conv4_block_drop3 	= nn.Dropout(0.5)  # dropout->(NEW)

		print(self.conv4_block_conv1)
		print(self.conv4_block_conv2)
		print(self.conv4_block_conv3)

		# initialize parameters and do require gradient computation
		for param in self.conv4_block_conv1.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_drop1.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_conv2.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_drop2.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_conv3.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_drop3.parameters():
		    param.requires_grad = True

		
		#-----------------
		# convolution block 5 : 512xW1xH1 -> 32xW2xH2 (to reduce the dimension)
		self.conv5_block_module = torch.nn.Sequential()
		self.conv5_block_module.add_module('conv_block5', nn.Conv2d(in_channels=512, out_channels=32, kernel_size=2, stride=1))
		self.conv5_block_module.add_module('relu_block5', nn.ReLU())
		#self.conv5_block_module.add_module('maxpool_block5', nn.MaxPool2d(kernel_size=2, stride=2))

		for param in self.conv5_block_module.parameters():
		    param.requires_grad = True

		# 'classifier' layer (fc-relu-drop): verify that the parameters are enabled for gradient computation
		for param in self.conv5_block_module.parameters():
			print("before fine-tuning conv5 layer parameters.requires_grad {} ".format(param.requires_grad))


		'''#-----------------
		# convolution block 6 : 512xW1xH1 -> 32xW2xH2 (to reduce the dimension)
		self.conv6_block_module = torch.nn.Sequential()
		self.conv6_block_module.add_module('conv_block6', nn.Conv2d(in_channels=128, out_channels=32, kernel_size=2, stride=1))
		self.conv6_block_module.add_module('relu_block6', nn.ReLU())
		#self.conv6_block_module.add_module('maxpool_block6', nn.MaxPool2d(kernel_size=2, stride=2))

		for param in self.conv6_block_module.parameters():
		    param.requires_grad = True

		# 'classifier' layer (fc-relu-drop): verify that the parameters are enabled for gradient computation
		for param in self.conv6_block_module.parameters():
			print("before fine-tuning conv6 layer parameters.requires_grad {} ".format(param.requires_grad))'''



		#-----------------
		# fully-connected layer 1: [512, 175, 350] (conv4_block) -> [1024] (nodes)
		# REZA: 1/23/19 (TOO MUCH CONNECTIONS (512*75*75) AFTER THE CONV4 LAYER)
		
		self.regress_module 		= torch.nn.Sequential()

		#self.regress_module.add_module('fc1', nn.Linear(512*75*75, 1024))
		self.regress_module.add_module('fc1', nn.Linear(32*74*74, 1024))
		self.regress_module.add_module('relu1', nn.ReLU())
		self.regress_module.add_module('drop1', nn.Dropout(0.5))


		# fully-connected layer 2: 1024 (nodes) -> 64 (nodes)
		self.regress_module.add_module('fc2', nn.Linear(1024, 256))
		self.regress_module.add_module('relu2', nn.ReLU())
		self.regress_module.add_module('drop2', nn.Dropout(0.5))

		# REZA: 07/03/19 created separate private variables to the last two inner-product layers

		# fully-connected layer 3: 256 (nodes) -> 64 (nodes)
		self.regress_module_fc3 	= nn.Linear(256, 64)
		self.regress_module_relu3 	= nn.ReLU()
		self.regress_module_drop3 	= nn.Dropout(0.5)


		# all the fully connected layer will be learned during backprop
		for param in self.regress_module.parameters():
		    param.requires_grad = True

		# redundant drop it later (by default requires_grad == True during instantiation)
		for param in self.regress_module_fc3.parameters():
		    param.requires_grad = True
		for param in self.regress_module_relu3.parameters():
		    param.requires_grad = True
		for param in self.regress_module_drop3.parameters():
		    param.requires_grad = True

		# 'classifier' layer (fc-relu-drop): verify that the parameters are enabled for gradient computation
		for param in self.regress_module.parameters():
			print("before fine-tuning fc layer parameters.requires_grad {} ".format(param.requires_grad))
		
		#pdb.set_trace()


	def forward(self, x):

		vis    = 0
		output = self.conv1_block(x)
		output = self.conv2_block(output)
		output = self.conv3_block(output)
		if (vis):
			print("feature map size() after conv1_3_block: {}".format(output.size()))

		output = self.conv4_block_conv1(output)
		output = self.conv4_block_drop1(output)
		output = self.conv4_block_conv2(output)
		output = self.conv4_block_drop2(output)
		output = self.conv4_block_conv3(output)
		output = self.conv4_block_drop3(output)
		if (vis):
			print("feature map size() after conv4_block: {}".format(output.size()))

		output = self.conv5_block_module(output)
		if (vis):
			print("feature map size() after conv5_block: {}".format(output.size()))

		'''output = self.conv6_block_module(output)
		if (vis):
			print("feature map size() after conv6_block: {}".format(output.size()))'''

		output = output.view(output.size(0),-1)
		output = self.regress_module(output)

		# additional codes to access the last fully connected layers
		output = self.regress_module_fc3(output)
		output = self.regress_module_relu3(output)
		output = self.regress_module_drop3(output)


		#output = output.view(output.size(0),-1)
		#output = output.squeeze()		
		if (vis):
			print("feature map size() after fc layers: {}".format(output.size()))

		return output



class EMNetv2(nn.Module):
	# network architecture:

	# input(1xWxH) -> conv features -> fully-connected 
	# 				   -> regression-loss and prediction

	def __init__(self, models):
		super(EMNetv2, self).__init__()

		model = models.vgg16(pretrained=True)
		model_features = list(model.features.children())		

		# first 3 convolutional blocks are used as it is
		conv1_block = model_features[0:5] 	# conv(64)->relu->conv(64)->relu->maxpool
		conv2_block = model_features[5:10] 	# conv(128)->relu->conv(128)->relu->maxpool
		conv3_block = model_features[10:16] # conv(256)->relu->conv(256)->relu->conv(256)->relu

		self.conv1_block = nn.Sequential(*conv1_block)
		self.conv2_block = nn.Sequential(*conv2_block)
		self.conv3_block = nn.Sequential(*conv3_block)

		print(self.conv1_block)
		print(self.conv2_block)
		print(self.conv3_block)

		# initialize parameters and do NOT require gradient computation
		for param in self.conv1_block.parameters():
		    param.requires_grad = False
		for param in self.conv2_block.parameters():
		    param.requires_grad = False
		for param in self.conv3_block.parameters():
		    param.requires_grad = False
		

		maxpool3 = model_features[16] 		# 3rd maxpool layer discarded (REDUNDANT BUT KEPT HERE FOR BETTER UNDERSTANDING)
		# fourth block is modified with dropout layers fused in (see below)
		conv4_block = model_features[17:22]

		# append all the blocks now
		conv4_block_conv1 		= conv4_block[0:2] # conv(512)->relu
		self.conv4_block_conv1 	= nn.Sequential(*conv4_block_conv1)
		self.conv4_block_drop1 	= nn.Dropout(0.5)  # dropout->(NEW)

		conv4_block_conv2 		= conv4_block[2:4] # conv(512)->relu
		self.conv4_block_conv2 	= nn.Sequential(*conv4_block_conv2)
		self.conv4_block_drop2 	= nn.Dropout(0.5)  # dropout->(NEW)

		conv4_block_conv3 		= conv4_block[4:6] # conv(512)->relu
		self.conv4_block_conv3  = nn.Sequential(*conv4_block_conv3)
		self.conv4_block_drop3 	= nn.Dropout(0.5)  # dropout->(NEW)

		print(self.conv4_block_conv1)
		print(self.conv4_block_conv2)
		print(self.conv4_block_conv3)

		# initialize parameters and do require gradient computation
		for param in self.conv4_block_conv1.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_drop1.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_conv2.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_drop2.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_conv3.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_drop3.parameters():
		    param.requires_grad = True

		
		#-----------------
		# convolution block 5 : 512xW1xH1 -> 32xW2xH2 (to reduce the dimension)
		self.conv5_block_module = torch.nn.Sequential()
		self.conv5_block_module.add_module('conv_block5', nn.Conv2d(in_channels=512, out_channels=32, kernel_size=2, stride=1))
		self.conv5_block_module.add_module('relu_block5', nn.ReLU())
		#self.conv5_block_module.add_module('maxpool_block5', nn.MaxPool2d(kernel_size=2, stride=2))

		for param in self.conv5_block_module.parameters():
		    param.requires_grad = True

		# 'classifier' layer (fc-relu-drop): verify that the parameters are enabled for gradient computation
		for param in self.conv5_block_module.parameters():
			print("before fine-tuning conv5 layer parameters.requires_grad {} ".format(param.requires_grad))


		'''#-----------------
		# convolution block 6 : 512xW1xH1 -> 32xW2xH2 (to reduce the dimension)
		self.conv6_block_module = torch.nn.Sequential()
		self.conv6_block_module.add_module('conv_block6', nn.Conv2d(in_channels=128, out_channels=32, kernel_size=2, stride=1))
		self.conv6_block_module.add_module('relu_block6', nn.ReLU())
		#self.conv6_block_module.add_module('maxpool_block6', nn.MaxPool2d(kernel_size=2, stride=2))

		for param in self.conv6_block_module.parameters():
		    param.requires_grad = True

		# 'classifier' layer (fc-relu-drop): verify that the parameters are enabled for gradient computation
		for param in self.conv6_block_module.parameters():
			print("before fine-tuning conv6 layer parameters.requires_grad {} ".format(param.requires_grad))'''



		#-----------------
		# fully-connected layer 1: [512, 175, 350] (conv4_block) -> [1024] (nodes)
		# REZA: 1/23/19 (TOO MUCH CONNECTIONS (512*75*75) AFTER THE CONV4 LAYER)
		
		self.regress_module = torch.nn.Sequential()

		#self.regress_module.add_module('fc1', nn.Linear(512*75*75, 1024))
		self.regress_module.add_module('fc1', nn.Linear(32*74*74, 1024))
		self.regress_module.add_module('relu1', nn.ReLU())
		self.regress_module.add_module('drop1', nn.Dropout(0.5))


		# fully-connected layer 2: 1024 (nodes) -> 64 (nodes)
		self.regress_module.add_module('fc2', nn.Linear(1024, 256))
		self.regress_module.add_module('relu2', nn.ReLU())
		self.regress_module.add_module('drop2', nn.Dropout(0.5))


		# fully-connected layer 3: 256 (nodes) -> 64 (nodes)
		self.regress_module.add_module('fc3', nn.Linear(256, 64))
		self.regress_module.add_module('relu3', nn.ReLU())
		self.regress_module.add_module('drop3', nn.Dropout(0.5))


		# fully-connected layer 4: 64 (nodes) -> 1 (node)
		self.regress_module.add_module('fc4', nn.Linear(64, 1))
		# there is no non-linear layer at the output layer.


		# all the fully connected layer will be learned during backprop
		for param in self.regress_module.parameters():
		    param.requires_grad = True

		# 'classifier' layer (fc-relu-drop): verify that the parameters are enabled for gradient computation
		for param in self.regress_module.parameters():
			print("before fine-tuning fc layer parameters.requires_grad {} ".format(param.requires_grad))
		
		#pdb.set_trace()


	def forward(self, x):

		vis = 0
		output = self.conv1_block(x)
		output = self.conv2_block(output)
		output = self.conv3_block(output)
		if (vis):
			print("feature map size() after conv1_3_block: {}".format(output.size()))

		output = self.conv4_block_conv1(output)
		output = self.conv4_block_drop1(output)
		output = self.conv4_block_conv2(output)
		output = self.conv4_block_drop2(output)
		output = self.conv4_block_conv3(output)
		output = self.conv4_block_drop3(output)
		if (vis):
			print("feature map size() after conv4_block: {}".format(output.size()))

		output = self.conv5_block_module(output)
		if (vis):
			print("feature map size() after conv5_block: {}".format(output.size()))

		'''output = self.conv6_block_module(output)
		if (vis):
			print("feature map size() after conv6_block: {}".format(output.size()))'''

		output = output.view(output.size(0),-1)
		output = self.regress_module(output)
		#output = output.view(output.size(0),-1)
		#output = output.squeeze()		
		if (vis):
			print("feature map size() after fc layers: {}".format(output.size()))

		return output


#-----------------------------------------------------------------------------
#---- network applied on 701x1401 pixels resolution
class EMNetv1(nn.Module):
	# network architecture:

	# input(1xWxH) -> conv features -> fully-connected 
	# 				   -> regression-loss and prediction

	def __init__(self, models):
		super(EMNetv1, self).__init__()

		model = models.vgg16(pretrained=True)
		model_features = list(model.features.children())

		# first 3 convolutional blocks are used as it is
		conv1_block = model_features[0:5] 	# conv(64)->relu->conv(64)->relu->maxpool
		conv2_block = model_features[5:10] 	# conv(128)->relu->conv(128)->relu->maxpool
		conv3_block = model_features[10:16] # conv(256)->relu->conv(256)->relu->conv(256)->relu

		self.conv1_block = nn.Sequential(*conv1_block)
		self.conv2_block = nn.Sequential(*conv2_block)
		self.conv3_block = nn.Sequential(*conv3_block)

		print(self.conv1_block)
		print(self.conv2_block)
		print(self.conv3_block)

		# initialize parameters and do NOT require gradient computation
		for param in self.conv1_block.parameters():
		    param.requires_grad = False
		for param in self.conv2_block.parameters():
		    param.requires_grad = False
		for param in self.conv3_block.parameters():
		    param.requires_grad = False
		

		maxpool3 = model_features[16] 		# 3rd maxpool layer discarded (REDUNDANT BUT KEPT HERE FOR BETTER UNDERSTANDING)
		# fourth block is modified with dropout layers fused in (see below)
		conv4_block = model_features[17:22]

		# append all the blocks now
		conv4_block_conv1 		= conv4_block[0:2] # conv(512)->relu
		self.conv4_block_conv1 	= nn.Sequential(*conv4_block_conv1)
		self.conv4_block_drop1 	= nn.Dropout(0.5)  # dropout->(NEW)

		conv4_block_conv2 		= conv4_block[2:4] # conv(512)->relu
		self.conv4_block_conv2 	= nn.Sequential(*conv4_block_conv2)
		self.conv4_block_drop2 	= nn.Dropout(0.5)  # dropout->(NEW)

		conv4_block_conv3 		= conv4_block[4:6] # conv(512)->relu
		self.conv4_block_conv3  = nn.Sequential(*conv4_block_conv3)
		self.conv4_block_drop3 	= nn.Dropout(0.5)  # dropout->(NEW)

		print(self.conv4_block_conv1)
		print(self.conv4_block_conv2)
		print(self.conv4_block_conv3)

		# initialize parameters and do require gradient computation
		for param in self.conv4_block_conv1.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_drop1.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_conv2.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_drop2.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_conv3.parameters():
		    param.requires_grad = True
		for param in self.conv4_block_drop3.parameters():
		    param.requires_grad = True


		#-----------------
		# convolution block 5 : 512xW1xH1 -> 32xW2xH2 (to reduce the dimension)
		self.conv5_block_module = torch.nn.Sequential()
		self.conv5_block_module.add_module('conv_block5', nn.Conv2d(in_channels=512, out_channels=128, kernel_size=2, stride=1))
		self.conv5_block_module.add_module('relu_block5', nn.ReLU())
		self.conv5_block_module.add_module('maxpool_block5', nn.MaxPool2d(kernel_size=2, stride=2))

		for param in self.conv5_block_module.parameters():
		    param.requires_grad = True

		# 'classifier' layer (fc-relu-drop): verify that the parameters are enabled for gradient computation
		for param in self.conv5_block_module.parameters():
			print("before fine-tuning conv5 layer parameters.requires_grad {} ".format(param.requires_grad))


		#-----------------
		# convolution block 6 : 512xW1xH1 -> 32xW2xH2 (to reduce the dimension)
		self.conv6_block_module = torch.nn.Sequential()
		self.conv6_block_module.add_module('conv_block6', nn.Conv2d(in_channels=128, out_channels=32, kernel_size=2, stride=1))
		self.conv6_block_module.add_module('relu_block6', nn.ReLU())
		self.conv6_block_module.add_module('maxpool_block6', nn.MaxPool2d(kernel_size=2, stride=2))

		for param in self.conv6_block_module.parameters():
		    param.requires_grad = True

		# 'classifier' layer (fc-relu-drop): verify that the parameters are enabled for gradient computation
		for param in self.conv6_block_module.parameters():
			print("before fine-tuning conv6 layer parameters.requires_grad {} ".format(param.requires_grad))

		# fully-connected layers of VGG (need to be retrained)
		'''mod = list(model.classifier.children())
		mod.pop()
		mod.append(nn.Linear(4096, 1))
		new_classifier = nn.Sequential(*mod)
		model_ft.classifier = new_classifier'''


		#-----------------
		# fully-connected layer 1: [512, 175, 350] (conv4_block) -> [32, 87, 174] (nodes)
		# REZA: 1/20/19 (TOO MUCH CONNECTIONS (512*175*350) AFTER THE CONV4 LAYER)
		
		# conv6 block 			 : 32*43*86  (conv5_block) -> 1024
		self.regress_module = torch.nn.Sequential()

		self.regress_module.add_module('fc1', nn.Linear(32*43*86, 1024))
		self.regress_module.add_module('relu1', nn.ReLU())
		self.regress_module.add_module('drop1', nn.Dropout(0.5))


		# fully-connected layer 2: 1024 (nodes) -> 64 (nodes)
		self.regress_module.add_module('fc2', nn.Linear(1024, 256))
		self.regress_module.add_module('relu2', nn.ReLU())
		self.regress_module.add_module('drop2', nn.Dropout(0.5))


		# fully-connected layer 3: 256 (nodes) -> 64 (nodes)
		self.regress_module.add_module('fc3', nn.Linear(256, 64))
		self.regress_module.add_module('relu3', nn.ReLU())
		self.regress_module.add_module('drop3', nn.Dropout(0.5))


		# fully-connected layer 4: 64 (nodes) -> 1 (node)
		self.regress_module.add_module('fc4', nn.Linear(64, 1))
		# there is no non-linear layer at the output layer.


		# all the fully connected layer will be learned during backprop
		for param in self.regress_module.parameters():
		    param.requires_grad = True

		# 'classifier' layer (fc-relu-drop): verify that the parameters are enabled for gradient computation
		for param in self.regress_module.parameters():
			print("before fine-tuning fc layer parameters.requires_grad {} ".format(param.requires_grad))
		
		#pdb.set_trace()


	'''def forward_shared(self, x):
		vis = 1


		output = self.conv1_block(x)
		output = self.conv2_block(output)
		output = self.conv3_block(output)
		if (vis):
			print("feature map size() after conv1_3_block: {}".format(output.size()))

		output = self.conv4_block_conv1(output)
		output = self.conv4_block_drop1(output)
		output = self.conv4_block_conv2(output)
		output = self.conv4_block_drop2(output)
		output = self.conv4_block_conv3(output)
		output = self.conv4_block_drop3(output)
		
		if (vis):
			print("feature map size() after conv4_block: {}".format(output.size()))

		return output
	'''


	def forward(self, x):

		vis = 0
		output = self.conv1_block(x)
		output = self.conv2_block(output)
		output = self.conv3_block(output)
		if (vis):
			print("feature map size() after conv1_3_block: {}".format(output.size()))

		output = self.conv4_block_conv1(output)
		output = self.conv4_block_drop1(output)
		output = self.conv4_block_conv2(output)
		output = self.conv4_block_drop2(output)
		output = self.conv4_block_conv3(output)
		output = self.conv4_block_drop3(output)
		if (vis):
			print("feature map size() after conv4_block: {}".format(output.size()))

		output = self.conv5_block_module(output)
		if (vis):
			print("feature map size() after conv5_block: {}".format(output.size()))

		output = self.conv6_block_module(output)
		if (vis):
			print("feature map size() after conv6_block: {}".format(output.size()))

		output = output.view(output.size(0),-1)
		output = self.regress_module(output)
		output = output.view(output.size(0),-1)
		if (vis):
			print("feature map size() after fc layers: {}".format(output.size()))
		

		return output

