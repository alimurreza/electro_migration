# this script train/test model with fusion of the two modalities
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
import argparse
from glob import glob
from pathlib import Path


# my classes
from network import *
from skimage.transform import pyramid_gaussian # multi-scale input


class EMFPDataPair(object):

	def __init__(self, dsets_images_a, dsets_images_b, failure_times, num_of_images, start=0):

		pil2tensor = transforms.ToTensor()
		Xa_scale0 	= []
		Xb_scale0 	= []
		Y 		= []
		#pdb.set_trace()
		vis_for_debug = 0
		for i in range(start, num_of_images):

			# get the CCD image
			cur_img_a, cur_label_a 	= dsets_images_a.__getitem__(i) 	# cur_img: instance of PIL object, cur_label: cropped_image(0) or uncropped_image(1)
			cur_path 		= dsets_images_a.imgs[i]
			cur_tensor 		= pil2tensor(cur_img_a) 		# (3, h, w) dim tensor
			a_scale0 		= cur_tensor 					
			a_scale0 		= a_scale0.numpy()			# ranges [0...1]

			# get the thermal image
			cur_img_b, cur_label_b 	= dsets_images_b.__getitem__(i) 	# cur_img: instance of PIL object, cur_label: cropped_image(0) or uncropped_image(1)
			cur_path 		= dsets_images_b.imgs[i]
			cur_tensor 		= pil2tensor(cur_img_b) 		# (3, h, w) dim tensor
			b_scale0 		= cur_tensor 					
			b_scale0 		= b_scale0.numpy()			# ranges [0...1]


			# pdb.set_trace()

			'''# sanity check the data with visualization (different modalities): REZA 06/10/19			
			cur_tensor_trans = cur_tensor.numpy().transpose(1,2,0)
			plt.figure()
			plt.imshow(cur_tensor_trans)
			plt.show()
			#pdb.set_trace()'''


			# gt failure times
			cur_ft 	= failure_times[i][0] 					# cur_ft is a failure time (in percentage)
			cur_ft 	= np.float32(cur_ft) 					# convert to Float32 instead of Double
					
			#pdb.set_trace()
			if (i%5 == 0):
				print("Electro Migration Failure Prediction: data generated for {} prediction {} ".format(i, cur_path[0]))
			#print("{}. label is {}".format(i, cur_ft))

			Xa_scale0.append(a_scale0) 	# value ranges [0,1] for CCD image
			Xb_scale0.append(b_scale0) 	# value ranges [0,1] for thermal image
			Y.append(cur_ft) 		# value ranges [0,100] since it will be used with 


		self.xa_scale0 		= Xa_scale0
		self.xb_scale0 		= Xb_scale0
		self.label 		= Y
		self.size 		= len(Xa_scale0)
		print("EMFPDataPair has {} elements".format(self.size))

	def __getitem__(self, index):
		return (self.xa_scale0[index], self.xb_scale0[index], self.label[index])

	def __len__(self):
		return self.size


def load_last_model():
	
	models 			= glob(model_path + '/*.pth')	
	all_epoch_losses 	= np.array([])	
	if models:
		#pdb.set_trace()
		model_ids = [(int(f.split('_')[-4]), f) for f in models]
		start_epoch, last_cp 	= max(model_ids, key=lambda item:item[0])
		print('Last checkpoint: ', last_cp)
		model.load_state_dict(torch.load(last_cp))
		all_losses 		= scipy.io.loadmat(model_path + '/Losses_epoch_{}'.format(start_epoch))		
		all_epoch_losses 	= all_losses['all_epoch_losses'][0]
	else:
		start_epoch = 0
		last_cp = ''
	#pdb.set_trace()
	return start_epoch, all_epoch_losses.tolist()




def resume_training():
	start_epoch, all_epoch_losses = load_last_model()
	#pdb.set_trace()
	for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
		epoch_loss = train_model(epoch)
		all_epoch_losses.append(epoch_loss)

		if (epoch%epoch_save_interval == 0):			
			torch.save(model.state_dict(), model_path + '/ModelEpoch_{}_Train_loss_{:.4f}.pth'.format(epoch, epoch_loss))
			scipy.io.savemat(model_path + '/Losses_epoch_{}'.format(epoch), {'all_epoch_losses': all_epoch_losses})
			#scipy.io.savemat(model_path + '/Losses_epoch_{}'.format(epoch), {'all_loss': all_loss, 'all_epoch_losses': all_epoch_losses})


def train_model(epoch):

	model.train(True) # this is a must step before training starts

	scheduler.step() # REZA: ADDRESS THIS WHEN MODEL IS LOADED AFTER A CERTAIN NUMBER OF EPOCHS
	running_loss = 0
	running_batch_count = 0

	for index, (xa_scale0, xb_scale0, classes) in enumerate(data_loader):
		if (is_cuda):
			input1, input2, classes = Variable(xa_scale0.cuda()), Variable(xb_scale0.cuda()), Variable(classes.cuda())
		else:
			input1, input2, classes = Variable(xa_scale0), Variable(xb_scale0), Variable(classes)

		optimizer.zero_grad()

		output1 = model(input1, input2)
		# Reza (01/19): DEBUG reshape the ground-truth and the prediction
		#output1 = output1.view(output1.size(0),-1)
		#classes = classes.view(classes.size(0),-1)
		#unsq_classes = torch.unsqueeze(classes, 1)

		#pdb.set_trace()

		loss = criterion(output1.squeeze(), classes.squeeze()) # MSELoss() or Smooth-L1 Loss()
		
		#print("batch{}. loss {}".format(index, loss.data[0]))
		#all_loss.append(loss.data[0])
		
		loss.backward()
		optimizer.step()

		running_loss += loss.data.item()
		running_batch_count = running_batch_count + batch_size

		if (index % 10 == 0):
			print("epoch {}/{} batch {}: loss computation {} ...".format(epoch, epochs, index, running_loss/running_batch_count))


	epoch_loss = running_loss / total_images

	for ll in range(50):
		print('{} Loss: {}'.format(epoch, epoch_loss))

	return epoch_loss


def test_model():

	model.eval() # this is a must step # otherwise batchnorm layer and dropout layer are in train mode by default
	preds = []
	labels = []

	for index, (xa_scale0, xb_scale0, classes) in enumerate(data_loader):
		if (is_cuda):
			input1, input2, classes = Variable(xa_scale0.cuda()), Variable(xb_scale0.cuda()), Variable(classes.cuda())
		else:
			input1, input2, classes = Variable(xa_scale0), Variable(xb_scale0), Variable(classes)

		print("Cuda enabled {}".format(is_cuda))
		#print("input1: shape of the inputs before feeding into model {}".format(input1.size()))
		# pdb.set_trace()

		output1 = model(input1, input2) 			# 2D feature repres. of two inputs
		
		print("batch {} ... ".format(index))

		tmp = output1.data.cpu()
		preds.append(tmp.numpy().tolist())


		tmp = classes.data.cpu()
		#tmp = tmp[0,:]
		labels.append(tmp.numpy().tolist())

                # output/emdatasetv1/emnetv1/*.mat
		#cur_file_name = format(index, '06d')
		#output_file_name        = output_path + cur_file_name + ".png"
		#tvutils.save_image(tmp, output_file_name)


		#pdb.set_trace()
		if (index % 10 == 0):
			print("saving {} prediction ".format(index))

	return preds, labels


#================================================================================================================================
# 						initialize user-provided variables

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch EMNet')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--device_id', type=int, default=0,
                    help='gpu device id (default: 1)')
parser.add_argument('--lr', type=float, default=0.000001,
                    help='learning rate for the adam optimizer')
parser.add_argument('--data_path', default='None',
                    help='data directory for train, val, and test. the root dir is in data/')
parser.add_argument('--output_path', default='None',
                    help='directory to save learned representation')
parser.add_argument('--model_path', default='None',
                    help='directory to save models and losses. root dir is in models/')
parser.add_argument('--network_name', default='emnetv2',
                    help='network type e.g., emnetv1 or emnetv2')
parser.add_argument('--dataset_name', default='electro_migra_002',
                    help='dataset name e.g., electro_migra_002, electro_migra_003 etc')
parser.add_argument('--epoch_save_interval', type=int, default=5,
                    help='model that would be saved after every given interval e.g.,  250')
parser.add_argument('--is_train', type=str2bool, nargs='?', const=True,
                    help='boolean variable indicating if we are training the model. for testing just disable this flag')
parser.add_argument('--model_name', default='None',
                    help='trained model name e.g., used during evaluation stage')
parser.add_argument('--eval_set', default='test',
                    help='which set to evaluate the model when is_train=False')
parser.add_argument('--fusion_method', default='concat',
                    help='fusion of features from two different modalities of input: i) concat, ii) element-wise multiplication, iii) TBD')


args 			= parser.parse_args()
batch_size 		= args.batch_size
epochs 			= args.epochs
lr 			= args.lr
data_path  		= args.data_path
output_path 		= args.output_path
network_name 		= args.network_name
model_path 		= args.model_path
dataset_name 		= args.dataset_name
epoch_save_interval 	= args.epoch_save_interval
is_train 	 	= args.is_train
eval_set 		= args.eval_set
model_name 		= args.model_name
fusion_method 		= args.fusion_method

#pdb.set_trace()
#------------------------------------------------------------------------------------------------
# create the directory for saving the trained models
model_dir = Path(model_path)
is_model_dir = model_dir.exists()
if (is_model_dir == 0):
    os.makedirs(model_dir)
else:
    print('model directory exits: {}'.format(model_dir))

# create the directory for saving the computed features using the trained model
output_file_path 	= output_path + eval_set
output_dir 		= Path(output_file_path)
is_output_dir 		= output_dir.exists()
if (is_output_dir == 0):
    os.makedirs(output_dir)
else:
    print('output directory exits: {}'.format(output_file_path))

#================================================================================================================================
# 						initialize global variables
args.cuda 			= True
device_id 			= args.device_id
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# select the pretrained model/scratch model
if (network_name == 'emnetv1'):
	model = EMNetv1(models)
	# Smooth L1 loss + Adam
	saved_model_name = 'emnetv5'
	saved_feat_name = 'emnetv5'

elif (network_name == 'emnetv2'):
	model = EMNetv2(models)
	#--------------------------------------------------------------------------------
	# REZA: experiment done until 01/2019 (emnet_v2_sml1_adam_v6 is the final result)
	'''# MSE loss + SGD
	#saved_model_name = 'emnetv2_mse_sgd'
	#saved_feat_name = 'emnetv2_mse_sgd'

	# Smooth L1 loss + SGD
	#saved_model_name = 'emnetv2_sml1_sgd'
	#saved_feat_name = 'emnetv2_sml1_sgd'

	# Smooth L1 loss + Adam
	saved_model_name = 'emnetv2_sml1_adam_v4'
	saved_feat_name = 'emnetv2_sml1_adam_v4'

	# Smooth L1 loss + Adam
	saved_model_name = 'emnetv2_sml1_adam_v5'
	saved_feat_name = 'emnet_v2_sml1_adam_v5'

	# Smooth L1 loss + Adam
	saved_model_name = 'emnetv2_sml1_adam_v6'
	saved_feat_name = 'emnetv2_sml1_adam_v6'

	# REZA: change this for testing on a different set
	saved_test_set_name = 'test'''
	#--------------------------------------------------------------------------------
	
	# Smooth L1 loss + Adam
	saved_feat_name  = 'emnetv2_feat'

elif (network_name == 'emnetv4'):
	model = EMNetv4(models)
	#pdb.set_trace()
	#--------------------------------------------------------------------------------	
	# Smooth L1 loss + Adam
	saved_feat_name  = 'emnetv4_feat'


elif (network_name== 'alexnet'):
	print("Network model for (model_name=alexnet) is not initialized ...")

# enable cudalabels
if (torch.cuda.is_available()):
	model = model.cuda()
	torch.cuda.set_device(device_id)
	is_cuda = True
else:
	is_cuda = False


# initialize the training hyper-parameters
momentum 	= 0.9
lr_step_size 	= epochs
criterion  	= torch.nn.SmoothL1Loss() # Reza (04/19): JUST A NOTE FOR YOURSELF - huber loss is less insensitive to outliers for regression
optimizer 	= torch.optim.Adam(model.parameters(), lr = lr)
scheduler 	= torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.1, last_epoch=-1)

#pdb.set_trace()

#================================================================================================================================
# 						data loading and preprocessing

'''# get the train + val file information
train_list 	= load_file(data_path + '/' + dataset_name + '/train_image_pairs.txt')	
val_list 	= load_file(data_path + '/' + dataset_name + '/val_image_pairs.txt')
#test_list 	= load_file(data_path + '/' + dataset_name + '/test_image_pairs.txt')'''

'''
config 		  	= {}
config['basedir'] 	= os.getcwd()

# [701, 1401] large image
#config['dataDir'] 	= 'data/v1'
#datasetName 	  	= 'emdatasetv1'

# [301, 301] small image
config['dataDir'] 	= 'data/v2'
datasetName 	  	= 'emdatasetv2'
'''

# load the failure-times for train and test data. labels['file_name'] has the image file corresponding to each failure time (not using though)

if (eval_set == 'train'):
	
	#pdb.set_trace()
	print("Processing for multi-modal inputs ...")	
	print("data_path ->{}".format(data_path + '/train_ccd'))
	dsets 			= {'train_image': datasets.ImageFolder(data_path + '/train_ccd', transform=None)}
	dsets_image_ccd 	= dsets['train_image']
	dsets 			= {'train_image': datasets.ImageFolder(data_path + '/train_thermal', transform=None)}
	dsets_image_thermal 	= dsets['train_image']
	#pdb.set_trace()

	labels 		= sio.loadmat(data_path + '/failure_times_train_sorted.mat')

elif (eval_set == 'test'):
	
	print("Processing for multi-modal image inputs is incomplete ...")
	print("data_path ->{}".format(data_path + '/test'))
	dsets 			= {'test_image': datasets.ImageFolder(data_path + '/test_ccd', transform=None)}
	dsets_image_ccd 	= dsets['test_image']
	dsets 			= {'test_image': datasets.ImageFolder(data_path + '/test_thermal', transform=None)}
	dsets_image_thermal 	= dsets['test_image']
	# pdb.set_trace()
	
	labels 		= sio.loadmat(data_path + '/failure_times_test_sorted.mat')
else:
	print('eval_set=Unknown')
	pdb.set_trace()

#pdb.set_trace()
labels 			= labels['failure_times']
total_images 		= len(dsets_image_ccd)
emfpdata 		= EMFPDataPair(dsets_image_ccd, dsets_image_thermal, labels, total_images, start=0)


if (is_train == True):	
	data_loader 	= torch.utils.data.DataLoader(emfpdata, batch_size=batch_size, shuffle=True)
	resume_training()	
	#pdb.set_trace()

else:	
	model_file_name = model_path + model_name
	model_file 	= Path(model_file_name)
	is_model_file 	= model_file.exists()
	if (is_model_file == 0):
		os.error('Error loading (model file does not exist): ' + model_file_name)
	else:
		print("Loading model {} ...".format(model_file_name))
		model.load_state_dict(torch.load(model_file_name))
	
	print("Evaluting the model on {} ...".format(eval_set))
	data_loader 	= torch.utils.data.DataLoader(emfpdata, batch_size=batch_size, shuffle=False) # don't shuffle. images will be processed sequentially
	predicted, gt 	= test_model()
	output_file_name = output_file_path + '/' + saved_feat_name + '_' + eval_set + '.mat'
	scipy.io.savemat(output_file_name, {'pred':predicted, 'gt':gt})



'''eval_feat_train_or_test = 0

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
		
		print("Forward pass on the test set ... ")
		predicted, gt = test_model(model_scratch, loader_test, is_cuda=True)
		saved_test_set_name = 'test'
		# output/emdatasetv1/emnetv1/*.mat		
		output_file_name 	= "output/" + datasetName + '/' + model_name  + "/"  + saved_feat_name + "_" + saved_test_set_name + ".mat"
		scipy.io.savemat(output_file_name, {'pred':predicted, 'gt':gt})'''
