function eval_prediction()
    
    % training images
    close all; clc;
    experiment             = 4;
    model_epoch_number     = 2500;
    init_coord_x        = [700, 750];
    init_coord_y        = [400, 450];
    cropped_img_size    = [300, 300]; % width, height
%     split               = 'test';
    split               = 'train';
    img_src_extension   = 'png';
    img_dest_extension  = 'png';
    vis                 = 0; if(vis)    figure; end;
    
%     src_dir = ['/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/data/v2/' split '_image/cropped_image'];
%     load(['/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/data/v2/failure_times_' split '.mat']);    
%     result = load(['/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/output/emdatasetv2/emnetv2/emnetv6_' split '.mat']);
    
% ------------------------------------------ experiment v1 ----------------
%     src_dir             = ['/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/data/v1/' split '_image/cropped_image'];
%     result = load(['/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/output/emdatasetv1/emnetv1/emnetv5_' split '.mat']);
%     load(['/Users/reza/Class_and_research/IU_research/pytorch/electro_migration_failure_pred/data/v1/failure_times_' split '.mat']);    


    if (experiment == 1)

    % ------------------------------------------ electro_migra_001  ----------------
        root_dir    = '/l/vision/v5/mdreza/electro_migration_failure_pred/';
        src_dir     = [root_dir '/data/v2/' split '/cropped_image'];
        load([root_dir '/data/v2/failure_times_' split '.mat']);    
        result = load([root_dir '/output/emdatasetv2/emnetv2/emnetv2_sml1_adam_v6_' split '.mat']);

    elseif (experiment == 2)
        % ----------------------------------------- electro_migra_002 -------------
        root_dir    = '/l/vision/v5/mdreza/electro_migration_failure_pred/';
        src_dir     = [root_dir '/data/electro_migra_002/' split '/cropped_image'];
        load([root_dir '/data/electro_migra_002/failure_times_' split '_sorted.mat']);    
        result      = load([root_dir '/output/output_emnetv2_electro_migra_002_lr.000001/' split '/emnetv2_feat_' split '.mat']);

    elseif (experiment == 3)
        % ----------------------------------------- electro_migra_003 -------------
        root_dir    = '/l/vision/v5/mdreza/electro_migration_failure_pred/';
        src_dir     = [root_dir '/data/electro_migra_003/' split '/cropped_image'];
        load([root_dir '/data/electro_migra_003/failure_times_' split '_sorted.mat']);    
        result      = load([root_dir '/output/output_emnetv2_electro_migra_003_lr.000001/' split '/emnetv2_feat_' split '.mat']);
    
    elseif (experiment == 4)
        % ----------------------------------------- (electro_migra_005 - electro_migra_025) CCD Image -------------
        root_dir    = '/l/vision/v5/mdreza/electro_migration_failure_pred/';
        src_dir     = [root_dir '/data/failure_at_void_v2/' split '_ccd/cropped_ccd'];
        load([root_dir '/data/failure_at_void_v2/failure_times_' split '_sorted.mat']);    
        result      = load([root_dir '/output/output_emnetv2_failure_at_void_v2_ccd_lr.000001/' split '/emnetv2_feat_' num2str(model_epoch_number) '_' split '.mat']);
        
    elseif (experiment == 5)
        % ----------------------------------------- (electro_migra_005 - electro_migra_025) thermal Image -------------
        root_dir    = '/l/vision/v5/mdreza/electro_migration_failure_pred/';
        src_dir     = [root_dir '/data/failure_at_void_v2/' split '_thermal/cropped_thermal'];
        load([root_dir '/data/failure_at_void_v2/failure_times_' split '_sorted.mat']);    
        result      = load([root_dir '/output/output_emnetv2_failure_at_void_v2_thermal_lr.000001/' split '/emnetv2_feat_' num2str(model_epoch_number) '_' split '.mat']);
        
    else

        
    end

    running_rmse = 0;
    running_mae = 0;
    running_sde = 0;
    
    ground_truth = result.gt;
    predicted = result.pred;
    diff = ground_truth - predicted;
    for iFiles=1:length(ground_truth)

        %% image visualizaition section (redundant)
        img_name    = file_names(iFiles,:);
        img_name    = img_name(1:end-7); % eliminate the '-01' postfix which got appended when I cropped a random [1401, 701] size from original. 
                                         % we are reusing the same train/test split but with a different crop from the original image.
        img         = imread(sprintf('%s/%s-01.%s', src_dir, img_name, img_src_extension));       
        if (vis)
            imagesc(img); title(img_name);
            pause;
            clf;
        end
%         imwrite(cropped_img, sprintf('%s/%s-01.%s',dest_dir, img_name, img_dest_extension));
        fprintf('%d) done processing %s ...%s\n',iFiles,split, img_name);
        
        %% evaluation code
        % rmse
        
        running_rmse = running_rmse + diff(iFiles)*diff(iFiles);
        
        % mae
        running_mae = running_mae + abs(diff(iFiles));
        
        % sde
%         tmp = 
        
    end
    
    RMSE = sqrt(running_rmse/length(ground_truth));
    MAE = running_mae/length(ground_truth);
    
    display(['RMSE: ' num2str(RMSE)]);
    display(['MAE: ' num2str(MAE)]);
%     save(['failure_times_' split '.mat'], 'failure_times', 'file_names');


b = bar([1:length(result.pred)], [ground_truth(1:length(result.pred)) predicted(1:length(result.pred))]); b(1).FaceColor = 'r'; b(2).FaceColor = 'g';
legend('gt-time', 'pred-time');
xlabel([split ' images']);
ylabel('life expired (in percentage)');
title(['RMSE: ' num2str(RMSE) ' and MAE: ' num2str(MAE)]);

end