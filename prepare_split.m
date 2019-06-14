% prepare the split the images in a dataset (experiments with failure happens at negative pad)

% Md Alimoor Reza
% mdreza@iu.edu
% Postdoctoral Associate, 
% Indiana University Bloomington
% January 2019


function prepare_split(dataset, global_par)

%     if (dataset == 1)
%         annotated_imgs_for_segmentation={'electro_migra_001_00005-01.png',...
%                                         'electro_migra_001_00065-01.png',...
%                                         'electro_migra_001_00075-01.png',...
%                                         'electro_migra_001_00085-01.png',...
%                                         'electro_migra_001_00095-01.png',...
%                                         'electro_migra_001_00105-01.png',...
%                                         'electro_migra_001_00115-01.png',...
%                                         'electro_migra_001_00125-01.png'};
%         include_annotated_imgs_for_segmentation = 1;
% %         src_dir = '/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/data/all_images/';
% %         dest_dir = '/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/data/';
%         
%     else
%         annotated_imgs_for_segmentation = {};
%         include_annotated_imgs_for_segmentation = 1;
%         
%     end

    root_dir        = global_par.root_dir;
    %root_dir        = '/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/data/';
    src_dir         = [root_dir '/data/electro_migra_' sprintf('%03d', dataset) '/'];
    dest_dir        = [root_dir '/data/electro_migra_' sprintf('%03d', dataset) '/'];
    %files = dir(sprintf('%s/*.png',src_dir));
    files           = dir(sprintf('%s/CCDImage/*.png',src_dir));
    
    %% the failure-times
    total_images    = length(files);
    failure_times   = 1:total_images;
    failure_times   = 100*failure_times/total_images;
    file_names      = {files.name};
    fnToftMap       = containers.Map(file_names, failure_times);
    
    %% train/val/test split
    training_indices = [];
%     if (include_annotated_imgs_for_segmentation)
%         for i=1:length(annotated_imgs_for_segmentation)
%             idx = find(strcmp(file_names, annotated_imgs_for_segmentation{i}));
%             training_indices = [training_indices idx];
%         end
%     end
   
    all_indices                 = 1:total_images;
    all_indices                 = setdiff(all_indices, training_indices);
    train_val_test_ratio        = global_par.train_val_test_ratio;
    %train_val_test_ratio        = [0.6 0.1 0.3];
    train_val_test_size         = round(train_val_test_ratio*length(all_indices));
    rand_indices                = randperm(length(all_indices));
    
    training_indices            = rand_indices(1:train_val_test_size(1));
    val_indices                 = rand_indices(train_val_test_size(1)+1:train_val_test_size(1)+train_val_test_size(2));  
    test_indices                = rand_indices(train_val_test_size(1)+train_val_test_size(2)+1:end);

    if ~isempty(setdiff(all_indices, [training_indices val_indices test_indices]))
        error('(train+val+test) size should be the same as all the images before split');
    end
    
%     keyboard;
    % training images
    make_split(training_indices, files, src_dir, dest_dir, 'png', 'png', 'train', fnToftMap);

    disp('done reshaping the dataset (train) ...');

    % validation images
    make_split(val_indices, files, src_dir, dest_dir, 'png', 'png', 'val', fnToftMap);

    disp('done reshaping the dataset (val) ...');

    % test images
    make_split(test_indices, files, src_dir, dest_dir, 'png', 'png', 'test', fnToftMap);

    disp('done reshaping the dataset (test) ...');

end