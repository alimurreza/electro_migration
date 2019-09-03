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

import sklearn.metrics as sklearn_metrics
# my classes
from network import *
from skimage.transform import pyramid_gaussian # multi-scale input


class EMFPData(object):

	def __init__(self, dsets_images, failure_times, num_of_images, start=0):

		pil2tensor = transforms.ToTensor()
		X_scale0 = []
		Y = []
		#pdb.set_trace()
		vis_for_debug = 0
		for i in range(start, num_of_images):
			cur_img, cur_label = dsets_images.__getitem__(i) 	# cur_img: instance of PIL object, cur_label: cropped_image(0) or uncropped_image(1)
			cur_path = dsets_images.imgs[i]
			#pdb.set_trace()
			cur_tensor = pil2tensor(cur_img) 			# (3, h, w) dim tensor
			scale0 = cur_tensor 					
			scale0 = scale0.numpy()					# ranges [0...1]

			# sanity check the data with visualization (different modalities): REZA 06/10/19			
			#cur_tensor_trans = cur_tensor.numpy().transpose(1,2,0)
			#plt.figure()
			#plt.imshow(cur_tensor_trans)
			#plt.show()
			#pdb.set_trace()

			# gt failure times
			#cur_ft 	= []
			cur_ft 	= failure_times[i]				# cur_ft is a failure time (in percentage)
			cur_ft 	= np.float32(cur_ft) 				# convert to Float32 instead of Double
			#cur_ft.append(tmp)
					
			
			if (i%5 == 0):
				print("Electro Migration Failure Prediction: data generated for {} prediction {} ".format(i, cur_path[0]))
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


def load_last_model():
	#pdb.set_trace()
	models_c 			= glob(model_path + '/*.pth')	
	train_all_epoch_losses 	= np.array([])
	val_all_epoch_losses = np.array([])
	#pdb.set_trace()
	if models_c:
		model_ids = [(int(f.split('_')[-7]), f) for f in models_c]
		start_epoch, last_cp 	= max(model_ids, key=lambda item:item[0])
		print('Last checkpoint: ', last_cp)
		model.load_state_dict(torch.load(last_cp))
		all_losses 		= scipy.io.loadmat(model_path + '/Losses_epoch_{}'.format(start_epoch))		
		train_all_epoch_losses 	= all_losses['train_all_epoch_losses'][0]
		val_all_epoch_losses 	= all_losses['val_all_epoch_losses'][0]
		
	else:
		start_epoch = 0
		last_cp = ''
	#pdb.set_trace()
	return start_epoch, train_all_epoch_losses.tolist(), val_all_epoch_losses.tolist()




def resume_training():
	start_epoch, train_all_epoch_losses, val_all_epoch_losses = load_last_model()
	#pdb.set_trace()
	for epoch in range(start_epoch + 1, epochs + 1):
		train_epoch_loss = train_model(epoch)
		val_epoch_loss = val_model(epoch)
		train_all_epoch_losses.append(train_epoch_loss)
		val_all_epoch_losses.append(val_epoch_loss)
		#pdb.set_trace()
		if (epoch%epoch_save_interval == 0):			
			torch.save(model.state_dict(), model_path + '/ModelEpoch_{}_Train_loss_{:.4f}_Val_loss_{:.4f}.pth'.format(epoch, train_epoch_loss,val_epoch_loss))
			scipy.io.savemat(model_path + '/Losses_epoch_{}'.format(epoch), {'train_all_epoch_losses': train_all_epoch_losses, 'val_all_epoch_losses':val_all_epoch_losses})
		#pdb.set_trace()

def train_model(epoch):

	model.train(True) # this is a must step before training starts

	scheduler.step() # REZA: ADDRESS THIS WHEN MODEL IS LOADED AFTER A CERTAIN NUMBER OF EPOCHS
	running_loss = 0
	running_batch_count = 0

	for index, (x_scale0, classes) in enumerate(data_loader):
		if (is_cuda):
			input1, classes = Variable(x_scale0.cuda()), Variable(classes.cuda())
		else:
			input1, classes = Variable(x_scale0), Variable(classes)

		optimizer.zero_grad()

		output1 = model(input1)
		# Reza (01/19): DEBUG reshape the ground-truth and the prediction
		#output1 = output1.view(output1.size(0),-1)
		#classes = classes.view(classes.size(0),-1)
		#unsq_classes = torch.unsqueeze(classes, 1);
		#pdb.set_trace()

		loss = criterion(output1.squeeze(), classes.squeeze()) # MSELoss() or Smooth-L1 Loss()
		
		#print("batch{}. loss {}".format(index, loss.data[0]))
		#all_loss.append(loss.data[0])
		
		loss.backward()
		optimizer.step()

		running_loss += loss.data.item()
		running_batch_count = running_batch_count + batch_size

		if (index % 10 == 0):
			print("epoch {}/{} batch {}: train loss computation {} ...".format(epoch, epochs, index, running_loss/running_batch_count))


	epoch_loss = running_loss / total_images

	for ll in range(50):
		print('{} Loss: {}'.format(epoch, epoch_loss))

	return epoch_loss


def val_model(epoch):
	model.eval()
	running_loss_v = 0
	running_batch_count_v = 0
	with torch.no_grad():
		for index, (images, classes) in enumerate(val_dataloader):
			if is_cuda:
				images, classes = Variable(images.cuda()), Variable(classes.cuda())
			
			output = model(images)

			loss = criterion(output.squeeze(),classes.squeeze())

			running_loss_v +=loss.data.item()
			running_batch_count_v = running_batch_count_v + batch_size

			if (index%10 == 0):
				print("epoch {}/{} batch {}: val loss {} ..".format(epoch,epochs, index, running_loss_v/running_batch_count_v))

	epoch_loss_v = running_loss_v / total_val_images

	return epoch_loss_v


def test_model():

	model.eval() # this is a must step # otherwise batchnorm layer and dropout layer are in train mode by default
	preds = []
	labels = []
	for index, (x_scale0, classes) in enumerate(data_loader):
		if (is_cuda):
			input1, classes = Variable(x_scale0.cuda()), Variable(classes.cuda())
		else:
			input1, classes = Variable(x_scale0), Variable(classes)

		
		#print("input1: shape of the inputs before feeding into model {}".format(input1.size()))
		# pdb.set_trace()

		output1 = model(input1) # 2D feature repres. of two inputs
		tmp = output1.data.cpu()
		preds.append(tmp.numpy().tolist())


		tmp = classes.data.cpu()
		#tmp = tmp[0,:]
		labels.append(tmp.numpy().tolist())

		#pdb.set_trace()
		if (index % 50 == 0):
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
parser.add_argument('--input_image_type', default='CCD',
                    help='Modality of the input: i) CCD, ii) Thermal, iii) Both')


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
input_image_type 	= args.input_image_type

#pdb.set_trace()
#------------------------------------------------------------------------------------------------
# create the directory for saving the trained models
model_dir = Path(model_path)
is_model_dir = model_dir.exists()
#pdb.set_trace()
if not (is_model_dir):
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
	#dsets 		= {'train_image': datasets.ImageFolder(data_path + '/train', transform=None)}
	#dsets_image 	= dsets['train_image']

	if (input_image_type == 'ccd'):
		print("data_path ->{}".format(data_path + '/train_ccd'))
		dsets 			= {'train_image': datasets.ImageFolder(data_path + '/train_ccd', transform=None)}
		dsets_image 		= dsets['train_image']

	elif(input_image_type == 'thermal'):
		print("data_path ->{}".format(data_path + '/train_thermal'))
		dsets 			= {'train_image': datasets.ImageFolder(data_path + '/train_thermal', transform=None)}
		dsets_image 		= dsets['train_image']

	elif(input_image_type == 'all'):
		print("data_path ->{}".format(data_path + '/train'))
		dsets 			= {'train_image': datasets.ImageFolder(data_path + '/train', transform=None)}
		dsets_image 		= dsets['train_image']

	else:
		dsets 			= {'train_image': datasets.ImageFolder(data_path + '/train_ccd', transform=None)}
		dsets_image_ccd 	= dsets['train_image']
		dsets 			= {'train_image': datasets.ImageFolder(data_path + '/train_thermal', transform=None)}
		dsets_image_thermal 	= dsets['train_image']
		print("Processing for multi-modal image inputs is incomplete ...")
		#pdb.set_trace()

	labels 		= sio.loadmat(data_path + '/failure_times_train_sorted.mat')

	## Validation Dataset images
	if (input_image_type == 'ccd'):
		print("data_path ->{}".format(data_path + '/val'))
		val_dsets			= {'val_image': datasets.ImageFolder(data_path + '/val_ccd', transform=None)}
		val_dsets_image 		= val_dsets['val_image']

	elif(input_image_type == 'thermal'):
		print("data_path ->{}".format(data_path + '/val_thermal'))
		val_dsets 			= {'val_image': datasets.ImageFolder(data_path + '/val_thermal', transform=None)}
		val_dsets_image 		= val_dsets['val_image']

	elif(input_image_type == 'all'):
		print("data_path ->{}".format(data_path + '/val'))
		val_dsets 			= {'val_image': datasets.ImageFolder(data_path + '/val', transform=None)}
		val_dsets_image 		= val_dsets['val_image']


	val_labels 		= sio.loadmat(data_path + '/failure_times_val_sorted.mat')
		

elif (eval_set == 'test'):
	
	if (input_image_type == 'ccd'):
		print("data_path ->{}".format(data_path + '/test'))
		dsets 			= {'test_image': datasets.ImageFolder(data_path + '/test_ccd', transform=None)}
		dsets_image 		= dsets['test_image']

	elif(input_image_type == 'thermal'):
		print("data_path ->{}".format(data_path + '/test_thermal'))
		dsets 			= {'test_image': datasets.ImageFolder(data_path + '/test_thermal', transform=None)}
		dsets_image 		= dsets['test_image']

	elif(input_image_type == 'all'):
		print("data_path ->{}".format(data_path + '/test'))
		dsets 			= {'test_image': datasets.ImageFolder(data_path + '/test', transform=None)}
		dsets_image 		= dsets['test_image']

	else:
		dsets 			= {'test_image': datasets.ImageFolder(data_path + '/test_ccd', transform=None)}
		dsets_image_ccd 	= dsets['test_image']
		dsets 			= {'test_image': datasets.ImageFolder(data_path + '/test_thermal', transform=None)}
		dsets_image_thermal 	= dsets['test_image']
		print("Processing for multi-modal image inputs is incomplete ...")
		#pdb.set_trace()

	
	#pdb.set_trace()
	labels 		= sio.loadmat(data_path + '/failure_times_test_sorted.mat')
else:
	print('eval_set=Unknown')
	#pdb.set_trace()


#pdb.set_trace()
labels 		= labels['failure_times'][0]
total_images 	= len(dsets_image)
print("total labels",len(labels))
pdb.set_trace()
emfpdata 	= EMFPData(dsets_image, labels, total_images, start=0)

#pdb.set_trace()

## Validation dataloader
if is_train:
	print("total val images",len(val_dsets_image))
	val_labels 		= val_labels['failure_times'][0]
	total_val_images 	= len(val_dsets_image)
	emfpdata_val 	= EMFPData(val_dsets_image, val_labels, total_val_images, start=0)
#pdb.set_trace()
## Train
if (is_train == True):	
	data_loader 	= torch.utils.data.DataLoader(emfpdata, batch_size=batch_size, shuffle=True)
	val_dataloader  = torch.utils.data.DataLoader(emfpdata_val, batch_size=batch_size, shuffle=False)
	resume_training()	
	#pdb.set_trace()

else:
	prd_array = []
	gt_array = []

	def compute_rmse(pred, gt):
		pred_array = []
		gt_array = []

		for ii in range(len(gt)):
			gt_array.append(gt[ii][0])
			pred_array.append(pred[ii][0][0])	

		rmse 			= np.sqrt(sklearn_metrics.mean_squared_error(gt_array, pred_array))
		mae 		 	= sklearn_metrics.mean_absolute_error(gt_array, pred_array)
		#pdb.set_trace()
		print("RMSE={}".format(rmse))
		print("MAE={}".format(mae))

		return rmse, mae


	## Test All Models
	all_models = glob(model_path + '/*.pth')
	all_epochs = []
	all_mae = []
	all_rmse = []
	best_rmse = 9999
	best_mae = 99999
	best_epoch = 0
	best_model = ''
	print("Evaluting the model on {} ...".format(eval_set))
	data_loader 	= torch.utils.data.DataLoader(emfpdata, batch_size=batch_size, shuffle=False) # don't shuffle. images will be processed sequentially	
	for m in all_models:
		model_epoch_num = m.split('_')[-7]
		
		m_name = os.path.basename(m)
		model_file_name = model_path + m_name
		model_file 	= Path(model_file_name)
		
		is_model_file 	= model_file.exists()
		if (is_model_file == 0):
			os.error('Error loading (model file does not exist): ' + model_file_name)
		else:
			print("Loading model {} ...".format(model_file_name))
			model.load_state_dict(torch.load(model_file_name))

		predicted, gt 	= test_model()
		output_file_name = output_file_path + '/' + saved_feat_name + '_' + model_epoch_num + '_' + eval_set + '.mat'
		scipy.io.savemat(output_file_name, {'pred':predicted, 'gt':gt})
		rmse, mae = compute_rmse(predicted, gt)

		if rmse < best_rmse:
			best_rmse = rmse
			best_mae = mae
			best_epoch = int(model_epoch_num)
			best_model = m_name
			print("best ", best_rmse, ' best_epoch', best_epoch)

		## epoch, rmse, mae
		all_epochs.append(int(model_epoch_num))
		all_mae.append(mae)
		all_rmse.append(rmse)
		


	scipy.io.savemat("save_results.mat",{'epochs':all_epochs, 'mae':all_mae, 'rmse':all_rmse, 'best_epoch':best_epoch,'best_mae':best_mae,'best_rmse':best_rmse})


