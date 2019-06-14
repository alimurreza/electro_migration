clear; clc;
% prepare-ground truth failure time
dataset         = 2;
if (dataset == 1)
    src_dir         = '/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/data/all_images/';
elseif(dataset == 2)
    src_dir         = '/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/data/electro_migra_002/';
else
    disp('NO DATASET DEFINED: ...');
    keyboard;
end

files           = dir(sprintf('%s/CCDImage/*.png',src_dir));
total_images    = length(files);
failure_time    = 1:total_images;
failure_time    = 100*failure_time/total_images;
file_names      = {files.name};
% csvwrite('failure_times.csv', failure_time);
% save('/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/data/electro_migra_002/failure_times.mat', 'failure_times', 'file_names');
