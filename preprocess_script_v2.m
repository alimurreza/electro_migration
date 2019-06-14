% steps for pre-processing
% this script prepares the train/test split for the 'contriction failure'
% first it prepares the 60%/10%/30% split on individual experiment (8B, 10T,
% etc). Each experiment requires separate cropping parameter since the
% experiment were done separately. once the split was done on individual
% experiment, it combines all the train together to prepare a single mat
% file of failure times. The images are also put into a separate directory.
% Similar setup is followed for the 'val' and 'test' partition.

%%
% *** Deep neural network model will only require this mat file and the train/val/test folder where the images are. ***
%%


% Md Alimoor Reza
% mdreza@iu.edu
% June, 2019

clear; close all; clc;

%% cropping on individual experiment
is_crop                                 = 0;
dataset_indices                         = 5:25;
global_par.split_names                  = {'train', 'val', 'test'};
global_par.train_val_test_ratio         = [0.6 0.1 0.3];
global_par.root_dir                     = '/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/';
global_par.root_dir                     = '/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/';
    
if (is_crop)
    for i=17:length(dataset_indices)
        iDataset    = dataset_indices(i);

        % find the train/val/test split
        prepare_split(iDataset, global_par);

        % find roi and randomly select a crop of size 301x301 pixels
        crop_images(iDataset, 1, global_par); % TRAIN
        crop_images(iDataset, 2, global_par); % VAL
        crop_images(iDataset, 3, global_par); % TEST

        % sort the file_names and failure_times
        sort_data(iDataset, 1, global_par); % TRAIN
        sort_data(iDataset, 2, global_par); % VAL
        sort_data(iDataset, 3, global_par); % TEST

        close all;

    end
    
end


% combine all frames from individual experiment and prepare the final train/val/test split for model learning/testing
% experiment_name     =  'within_coil_failure_experiment_v1';
experiment_name       = 'failure_at_void_v3';

if ~exist([global_par.root_dir experiment_name])
    mkdir([global_par.root_dir '/data/' experiment_name]);
end

for split_id=1:3
    
    all_file_names      = [];
    all_failure_times   = [];

    for i=1:length(dataset_indices)
        
        dataset             = dataset_indices(i);     
        split               = global_par.split_names{split_id};
        
        src_ccd_dir         = [global_par.root_dir '/data/electro_migra_' sprintf('%03d',dataset) '/' split '/cropped_ccd/'];
        src_thermal_dir     = [global_par.root_dir '/data/electro_migra_' sprintf('%03d',dataset) '/' split '/cropped_thermal/'];
        
        dest_ccd_dir        = [global_par.root_dir '/data/' experiment_name '/' split '/cropped_ccd/'];
        dest_thermal_dir    = [global_par.root_dir '/data/' experiment_name '/' split '/cropped_thermal/'];
        
        load(                 [global_par.root_dir '/data/electro_migra_' sprintf('%03d',dataset) '/failure_times_' split '_sorted.mat'] );    

        if ~exist(dest_ccd_dir)
            mkdir(dest_ccd_dir);
        end
        
        if ~exist(dest_thermal_dir)
            mkdir(dest_thermal_dir);
        end
        
        
        % add file names  
        for j=1:size(file_names,1)
            all_file_names  = [all_file_names; file_names(j,:)];
            img_name        = file_names(j,:);
            img_ccd         = imread([src_ccd_dir '/' img_name]);
            imwrite(img_ccd, sprintf('%s/%s',dest_ccd_dir, img_name));
            img_thermal     = imread([src_thermal_dir '/' img_name]);
            imwrite(img_thermal, sprintf('%s/%s', dest_thermal_dir, img_name));
            
        end
        
        % add failure times
        all_failure_times   = [all_failure_times; failure_times];
        clear file_names failure_times;
        
        
        
        disp(['Done processing ' split ' - electro_migra_' sprintf('%03d',dataset)]);
        
    end
    
    % save the file_names and failure_times for the split
    file_names      = all_file_names;
    failure_times   = all_failure_times;
    save([global_par.root_dir '/data/' experiment_name '/failure_times_' split '_sorted.mat'], 'file_names', 'failure_times');
    clear file_names failure_times;
end
