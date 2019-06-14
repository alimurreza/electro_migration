function sort_data(dataset, split_id, global_par)
% this script sorts the name of the files and puts the associated failure times for the final regression problem

% Md Alimoor Reza
% mdreza@iu.edu
% Postdoctoral Associate, 
% Indiana University Bloomington
% January 2019

    % sort the image names
    split_names = global_par.split_names;
    split       = split_names{split_id};

    if (dataset > 0)
%         load(['/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/data/v2/failure_times_' split '_unsorted.mat']);    
%         src_dir = '/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/data/v2/selected_images_test/';    
%     elseif(dataset == 2)
%         src_dir  = '/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/data/electro_migra_002/';
%         load([src_dir '/failure_times_' split '_unsorted.mat']);    
% 
%     elseif(dataset == 3)
        root_dir    = global_par.root_dir;
        src_dir     = [root_dir '/data/electro_migra_' sprintf('%03d',dataset) '/'];
        load([src_dir '/failure_times_' split '_unsorted.mat']); 

    else
        disp('NO DATASET DEFINED: ...');
        keyboard;

    end

    all_files = [];
    for i=1:length(file_names)

        all_files = [all_files; file_names{i}]

    end

    [sorted_files, idx] = sortrows(all_files);
    sorted_fts = failure_times(idx);
    file_names = sorted_files;
    failure_times = sorted_fts;
    % save('failure_times_train.mat', 'file_names', 'failure_times');
    save([src_dir '/failure_times_' split '_sorted.mat'], 'file_names', 'failure_times');
end