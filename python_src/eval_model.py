import os
import os.path as osp
import scipy.io as sio
import sklearn.metrics as sklearn_metrics
import pdb

config 		  	= {}
config['basedir'] 	= os.getcwd()
config['outDir'] 	= 'output'
config['datasetName'] 	= 'emdatasetv2'
config['netName'] 	= 'emnetv2'

result_dir = osp.join(config['basedir'], config['outDir'], config['datasetName'], config['netName'])

results 		= sio.loadmat(result_dir + '/emnetv6_.mat')
gt 			= results['gt']
pred 			= results['pred']

pred_array = []
gt_array = []

for ii in range(len(gt)):
	gt_array.append(gt[ii][0])
	pred_array.append(pred[ii][0][0])
	
#pdb.set_trace()

rmse 			= sklearn_metrics.mean_squared_error(gt_array, pred_array)
mae 		 	= sklearn_metrics.mean_absolute_error(gt_array, pred_array)

print("RMSE={}".format(rmse))
print("MAE={}".format(mae))



