import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import models,datasets
import torchvision.utils as tvutils
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io
from pathlib import Path
import time
import pdb
import os.path as osp
import os
import scipy.io as sio



# my classes
from network import *
#import data_loader as dl
#import util as my_util
from skimage.transform import pyramid_gaussian # multi-scale input


class EMFPData(object):

	def __init__(self, dsets_images, failure_times, num_of_images, start=0):

		pil2tensor = transforms.ToTensor()
		X_scale0 = []
		Y = []

		vis_for_debug = 0
		for i in range(start, num_of_images):
			cur_img, cur_label = dsets_images.__getitem__(i) 	# cur_img: is an instance of PIL object, cur_label: class label
			#pdb.set_trace()
			cur_tensor = pil2tensor(cur_img) 			# (3, h, w) dim tensor
			scale0 = cur_tensor 					
			scale0 = scale0.numpy()					# ranges [0...1]

			# gt failure times
			#cur_ft 	= []
			cur_ft 	= failure_times[i][0] 				# cur_ft is a failure time (in percentage)
			cur_ft 	= np.float32(cur_ft) 				# convert to Float32 instead of Double
			#cur_ft.append(tmp)
					
			#pdb.set_trace()
			#if (i%5 == 0):
			#	print("Electro Migration Failure Prediction data generated for {} prediction ".format(i))

			#print("{}. label is {}".format(i, cur_ft))

			X_scale0.append(scale0) 	# value ranges [0,1]
			Y.append(cur_ft) 		# value ranges [0,100] since it will be used with 


		self.x_scale0 		= X_scale0
		self.label 		= Y
		self.size 		= len(X_scale0)
		print("EMFPData has {} elements".format(self.size))

	def __getitem__(self, index):
		return (self.x_scale0[index], self.label[index])

	def __len__(self):
		return self.size

def train_model(model, data_loader, batch_size, scheduler=None, size=10, epochs=20,optimizer=None, is_cuda=False):

	all_epoch_losses = []	
	all_loss = []

	for epoch in range(epochs):

		scheduler.step()
		running_loss = 0
		running_batch_count = 0

		for index, (x_scale0, classes) in enumerate(data_loader):
			if (is_cuda):
				input1, classes = Variable(x_scale0.cuda()), Variable(classes.cuda())
			else:
				input1, classes = Variable(x_scale0), Variable(classes)

			#print("Cuda enabled {}".format(is_cuda))

			output1 = model(input1)


			# reshape the ground-truth and the prediction
			#output1 = output1.view(output1.size(0),-1)
			#classes = classes.view(classes.size(0),-1)
			#unsq_classes = torch.unsqueeze(classes, 1);
			#pdb.set_trace()

			loss = criterion(output1.squeeze(), classes.squeeze()) # MSELoss() or Smooth-L1 Loss()
			#loss = criterion(output1, unsq_classes)

			#print("batch{}. loss {}".format(index, loss.data[0]))
			all_loss.append(loss.data[0])

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			running_loss += loss.data[0]
			running_batch_count = running_batch_count + batch_size

			#if (index % 1000 == 0):
			print("epoch {}/{} batch {}: loss computation {} ...".format(epoch, epochs, index, running_loss/running_batch_count))

		        ##print("training ... backpropagation done")

		epoch_loss = running_loss / size
		all_epoch_losses.append(epoch_loss)
		#epoch_acc =  float(correct)/float(total)
		for ll in range(50):
			print('{} Loss: {}'.format(epoch, epoch_loss))

	return all_loss, all_epoch_losses



def test_model(model, data_loader, is_cuda=False):
#def test_model(model, data_loader, output_path, is_cuda=False):

	#model.eval() # this is a must step # otherwise batchnorm layer and dropout layer are in train mode by default
	preds = []
	labels = []
	for index, (x_scale0, classes) in enumerate(data_loader):
		if (is_cuda):
			input1, classes = Variable(x_scale0.cuda()), Variable(classes.cuda())
		else:
			input1, classes = Variable(x_scale0), Variable(classes)

		print("Cuda enabled {}".format(is_cuda))
		#print("input1: shape of the inputs before feeding into model {}".format(input1.size()))
		#pdb.set_trace()

		output1 = model(input1) # 2D feature repres. of two inputs
		
		print("batch {} ... ".format(index))

		tmp = output1.data.cpu()
		preds.append(tmp.numpy().tolist())


		tmp = classes.data.cpu()
		#tmp = tmp[0,:]
		labels.append(tmp.numpy().tolist())

                # output/emdatasetv1/emnetv1/*.mat
		'''cur_file_name = format(index, '06d')
		output_file_name        = output_path + cur_file_name + ".png"
		tvutils.save_image(tmp, output_file_name)'''


		#pdb.set_trace()
		if (index % 10 == 0):
			print("saving {} prediction ".format(index))




	return preds, labels

'''
# create pyramid gaussian for three different scale of an input image
def preprocess_train_or_test_set(dsets_train_image,dsets_train_gt, num_of_images, start=0):


	pil2tensor = transforms.ToTensor()
	#pil2transform = transforms.ToTensor()

	X_scale0_train = []
	X_scale1_train = []
	X_scale2_train = []
	Y_train = []

	vis_for_debug = 0
	#for i in range(len(dsets_train_image)):
	for i in range(start, num_of_images):
		cur_img, cur_label = dsets_train_image.__getitem__(i) # cur_img is an instance of PIL object
		cur_tensor = pil2tensor(cur_img) # (3, h, w) dim tensor (but pyramid_gaussian() would scale on the first two dimensions)
		scale0 = cur_tensor 	# already ranges between [0,1] 
		scale0 = scale0*255.0
		scale0 = scale0.numpy()
		cur_numpy_trans = cur_tensor.numpy().transpose(1,2,0) # turn into numpy array (h, w, 3)
		dst = tuple(pyramid_gaussian(cur_numpy_trans, max_layer=2, downscale=2)) # cur_numpy_trans already has [0,1] no need to normalize ( divide by 255.0

		# pyramid level-1
		scale1_trans = dst[1].transpose(2,0,1) # convert back to (3, h, w) tensor format
		scale1 = scale1_trans*255. # 

		#pdb.set_trace()

		if (vis_for_debug):
			plt.figure
			plt.imshow(dst[1]*255)
			plt.show()
			#pdb.set_trace()

		# pyramid level-2
		scale2_trans = dst[2].transpose(2,0,1) # convert back to (3, h, w) tensor format
		scale2 = scale2_trans*255. # 

		if (vis_for_debug):
			plt.figure
			plt.imshow(dst[2]*255.)
			plt.show()

		# gt image
		cur_gt, cur_label = dsets_train_gt.__getitem__(i) # cur_gt is an instance of PIL object
		cur_tensor = pil2tensor(cur_gt) # (3, h, w) dim tensor
		x, h, w = cur_tensor.size()
		binary_gt = torch.empty(1, 1, h, w, dtype=torch.float)
		binary_gt[0,0,:,:] = cur_tensor[0,:,:]
		#pdb.set_trace()

		#print("scale1 (NumpyType)", scale0.dtype)
		##scale1 = scale1.type(torch.FloatTensor)
		scale1 = np.float32(scale1)
		#print("scale1 (NumpyType)", scale1.dtype)
		scale2 = np.float32(scale2)
		#print("scale2 (NumpyType)", scale2.dtype)

		#pdb.set_trace()
		if (i%100 == 0):
			print("Gaussian pyramid generated for {} prediction ".format(i))

		X_scale0_train.append(scale0) 	# value ranges [0,255]
		X_scale1_train.append(scale1) 	# value ranges [0,255]
		X_scale2_train.append(scale2) 	# value ranges [0,255]
		Y_train.append(binary_gt) 	# value ranges [0,1] since it will be used with binary-cross-entropy loss function

	return X_scale0_train, X_scale1_train, X_scale2_train, Y_train

'''


#------------------------------------------------------------------------------------------------
# INITIALIZE THE MODEL
#pdb.set_trace()
use_gpu 	= torch.cuda.is_available()
if (use_gpu == True):
	is_cuda = True
	torch.cuda.set_device(1)
else:
	is_cuda = False

model_name 	= 'emnetv2'
# select the pretrained model/scratch model
if (model_name == 'emnetv1'):
	model_scratch = EMNetv1(models)
	#saved_model_name = 'emnetv1'
	#saved_feat_name = 'emnetv1'
	
	# MSE loss + SGD
	#saved_model_name = 'emnetv2'
	#saved_feat_name = 'emnetv2'

	# Smooth L1 loss + SGD
	#saved_model_name = 'emnetv3'
	#saved_feat_name = 'emnetv3'

	# Smooth L1 loss + Adam
	#saved_model_name = 'emnetv4'
	#saved_feat_name = 'emnetv4'

	# Smooth L1 loss + Adam
	saved_model_name = 'emnetv5'
	saved_feat_name = 'emnetv5'


	# REZA: change this for testing on a different set
	saved_test_set_name = ''

elif (model_name == 'emnetv2'):
	model_scratch = EMNetv2(models)
	#saved_model_name = 'emnetv1'
	#saved_feat_name = 'emnetv1'
	
	# MSE loss + SGD
	#saved_model_name = 'emnetv2'
	#saved_feat_name = 'emnetv2'

	# Smooth L1 loss + SGD
	#saved_model_name = 'emnetv3'
	#saved_feat_name = 'emnetv3'

	# Smooth L1 loss + Adam
	saved_model_name = 'emnetv4'
	saved_feat_name = 'emnetv4'

	# Smooth L1 loss + Adam
	saved_model_name = 'emnetv5'
	saved_feat_name = 'emnetv5'

	# Smooth L1 loss + Adam
	saved_model_name = 'emnetv6'
	saved_feat_name = 'emnetv6'

	# Smooth L1 loss + Adam
	saved_model_name = 'emnetvDUMMY'
	saved_feat_name = 'emnetvDUMMY'

	# REZA: change this for testing on a different set
	saved_test_set_name = 'test'

elif (model_name == 'alexnet'):
	#models_scratch = SiameseAlexNetv1(models)
	#saved_feat_dir = 'alexnet_conv'
	#saved_model_name = 'fine_tuned_alexnet'
	#saved_feat_name = 'alexnet_feat'
	print("Network model for (model_name=alexnet) is not initialized ...")

# enable cuda library if available
if (use_gpu == True):
    model_scratch = model_scratch.cuda()

batch_size = 1
# initialize the training hyper-parameters
'''# MSELoss parameters with SGD
#lr 			= 0.000001 	# diverges (with MSELoss)
#lr 			= 0.0000001 	# diverges after 10th epoch (with MSELoss)
lr 			= 0.00000001
epoch 			= 100
lr_step_size 		= 10
criterion 		= torch.nn.MSELoss()
# SGD optimizer
momentum 		= 0.9 # Reza: 1/22/2019 try chaning it to 0.3 or 0.5 or 0.7 and see if that helps
optimizer 		= torch.optim.SGD(model_scratch.parameters(), lr = lr, momentum=momentum)'''



'''# SmoothL1Loss parameters with SGD
#lr 			= 0.00000001 # loss gets stuck at 50.54
#lr 			= 0.0000001  # very slowly moving back from 50.36 -> 50.33 -> 50.30 ...
lr 			= 0.000001   # gets stuck at loss 26.0
epoch 			= 50
lr_step_size 		= 25
criterion 	 	= torch.nn.SmoothL1Loss()
# SGD optimizer
momentum 		= 0.9 # Reza: 1/22/2019 try chaning it to 0.3 or 0.5 or 0.7 and see if that helps
optimizer 		= torch.optim.SGD(model_scratch.parameters(), lr = lr, momentum=momentum)
scheduler1 		= torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.1, last_epoch=-1)'''


'''
# RMSProp Optimizer parameters
lr 			= 0.001 #
epsilon 		= 1e-08
epoch 			= 50
lr_step_size 		= epoch/8
optimizer 		= torch.optim.RMSprop(model_scratch.parameters(), lr=lr, alpha=0.90, eps=epsilon)
scheduler1 		= torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.1, last_epoch=-1)'''


# ADAM Optimizer parameters
lr 		= 0.000001 #
momentum 	= 0.9
epoch 		= 8000
lr_step_size 	= 8000
criterion  	= torch.nn.SmoothL1Loss()
optimizer 	= torch.optim.Adam(model_scratch.parameters(), lr = lr)
scheduler1 	= torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.1, last_epoch=-1)




#----------------------------------------------------------------------
# initialize the dataset parameters and preprocess
# STEP 1: prepare data and labels
'''
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

prep_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        '''


config 		  	= {}
config['basedir'] 	= os.getcwd()

# [701, 1401] large image
#config['dataDir'] 	= 'data/v1'
#datasetName 	  	= 'emdatasetv1'

# [301, 301] small image
config['dataDir'] 	= 'data/v2'
datasetName 	  	= 'emdatasetv2'
data_root_train   	= osp.join(config['basedir'], config['dataDir'])
data_root_test 	  	= osp.join(config['basedir'], config['dataDir'])

print("datapath ->{}".format(data_root_train + '/train_image'))
dsets_train ={'train_image': datasets.ImageFolder(data_root_train + '/train_image', transform=None)}

# REZA: change this for testing on a different set
dsets_test ={'test_image': datasets.ImageFolder(data_root_test + '/test_image', transform=None)}

dsets_train_image 	= dsets_train['train_image']
dsets_test_image 	= dsets_test['test_image']

#pdb.set_trace()
# load the failure-times for train and test data
# train_labels['file_name'] has the image file corresponding to each failure time (not using)
train_labels 		= sio.loadmat(data_root_train + '/failure_times_train.mat')
test_labels 		= sio.loadmat(data_root_train + '/failure_times_test.mat')
train_labels 		= train_labels['failure_times']
test_labels 		= test_labels['failure_times']
total_images_train 	= len(dsets_train_image)
total_images_test   	= len(dsets_test_image)

emfpdata_train = EMFPData(dsets_train_image, train_labels, total_images_train, start=0)
emfpdata_test  = EMFPData(dsets_test_image,  test_labels,  total_images_test,  start=0)

#pdb.set_trace()

'''print("Loading the triplet data into the dataloaders (train) ...")
X_scale0_train, X_scale1_train, X_scale2_train, Y_train = preprocess_train_or_test_set(dsets_train_image, dsets_train_gt, 15, start=0)
tripletData_train = TripletData(X_scale0_train, X_scale1_train, X_scale2_train, Y_train)

total_images = len(dsets_test_image)
print("Loading the triplet data into the dataloaders (test) ... %d ".format(total_images))
X_scale0_test, X_scale1_test, X_scale2_test, Y_test = preprocess_train_or_test_set(dsets_test_image, dsets_test_gt, total_images, start=4349)
tripletData_test = TripletData(X_scale0_test, X_scale1_test, X_scale2_test, Y_test)'''

#pdb.set_trace()



eval_feat_train_or_test = 0

# models/emdatasetv2
model_dir_name = 'models/' + datasetName
model_dir = Path(model_dir_name)
is_model_dir = model_dir.exists()
#pdb.set_trace()
if (is_model_dir == 0):
	os.mkdir(model_dir)
	print("created model directory ... {}".format(model_dir))

model_file_name = 'models/' + datasetName + '/' + saved_model_name + '.pth'
model_file = Path(model_file_name)
is_model_file = model_file.exists()

if (is_model_file == 0):
	model_scratch.train(True)
	loader_train = torch.utils.data.DataLoader(emfpdata_train, batch_size=batch_size, shuffle=True)
	print("Training the model ... ")
	t0 = time.time()
	all_loss, all_epoch_losses = train_model(model_scratch, data_loader=loader_train, batch_size=batch_size, scheduler=scheduler1, size=total_images_train, epochs=epoch,optimizer=optimizer, is_cuda=is_cuda)
	
	#pdb.set_trace()

	print("Training time {}".format(time.time()-t0))
	torch.save(model_scratch, model_file_name)
	print("Saving the trained model {} ".format(model_file_name))
	# 'models/' + datasetName + '/' +
	output_file_name = 'models/' + datasetName + '/' + saved_model_name + "_loss.mat"
	scipy.io.savemat(output_file_name, {'all_loss': all_loss, 'all_epoch_losses': all_epoch_losses})
else:

	print("Loading model {} ...".format(model_file_name))
	model_scratch = torch.load(model_file_name)
	model_scratch.eval() # otherwise batchnorm layer and dropout layer are in train mode by default
	if (use_gpu == True):
	    model_scratch = model_scratch.cuda()

	if (eval_feat_train_or_test == 1):

		print("Evaluting the model on the train set ...")
		loader_train = torch.utils.data.DataLoader(emfpdata_train, batch_size=batch_size, shuffle=False)
		print("Forward pass on the train set ... ")
		predicted, gt = test_model(model_scratch, loader_train, is_cuda=True)
		saved_train_set_name = 'train'
		# output/emdatasetv2/emnetv2/*.mat
		output_dir_name = "output/" + datasetName + '/' + model_name
		output_dir = Path(output_dir_name)
		is_output_dir = output_dir.exists()
		if (is_output_dir == 0):
			os.makedirs(output_dir)
			print("created output directory ... {}".format(is_output_dir))			
		output_file_name 	= "output/" + datasetName + '/' + model_name  + "/"  + saved_feat_name + "_" + saved_train_set_name + ".mat"
		scipy.io.savemat(output_file_name, {'pred':predicted, 'gt':gt})

	else:

		print("Evaluting the model on the test set ...")
		loader_test = torch.utils.data.DataLoader(emfpdata_test, batch_size=batch_size, shuffle=False)
		'''# output/piano_v1/fgsegnet_v1/test1-mp4.raw/
		output_path = "output/" + datasetName + '/' + model_name  + "/"  + saved_test_set_name + "/"'''

		print("Forward pass on the test set ... ")
		predicted, gt = test_model(model_scratch, loader_test, is_cuda=True)
		saved_test_set_name = 'test'
		# output/emdatasetv1/emnetv1/*.mat		
		output_file_name 	= "output/" + datasetName + '/' + model_name  + "/"  + saved_feat_name + "_" + saved_test_set_name + ".mat"
		scipy.io.savemat(output_file_name, {'pred':predicted, 'gt':gt})


