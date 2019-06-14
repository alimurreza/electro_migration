% steps for pre-processing
% this script prepares the train/val/test split for the 'failure at negative pad' first it prepares the 60%/10%/30% split on an individual
% experiment. Each experiment requires separate cropping parameter since the experiment were done separately. The images are also 
% put into a separate directory. Similar setup is followed for the 'test' partition.

%%
% *** Deep neural network model will only require this mat file and the train/val/test folder where the images are. ***
%%



% Md Alimoor Reza
% mdreza@iu.edu
% Postdoctoral Associate, 
% Indiana University Bloomington
% January 2019


clear; close all; clc;


is_crop                                 = 1;
dataset_indices                         = 1:3;
global_par.split_names                  = {'train', 'val', 'test'};
global_par.train_val_test_ratio         = [0.6 0.1 0.3];
global_par.root_dir                     = '/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/';
    
    
if (is_crop)
    for i=2:3
        iDataset    = dataset_indices(i);

        % find the train/val/test split
        prepare_split_v1(iDataset, global_par);

        % find roi and randomly select a crop of size 301x301 pixels
        crop_images(iDataset, 1, global_par); % TRAIN
        crop_images(iDataset, 2, global_par); % VAL
        crop_images(iDataset, 3, global_par); % VAL

        % sort the file_names and failure_times
        sort_data(iDataset, 1, global_par);     % TRAIN
        sort_data(iDataset, 2, global_par);     % VAL
        sort_data(iDataset, 3, global_par);     % TEST

        close all;

    end
    
end