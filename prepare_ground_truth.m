%% this script crops the electro-migra images

clc; clear;

% gt_coord_x = [465, 465+950 ];
% gt_coord_y = [577, 577+130];

init_coord_x = [1, 300];
init_coord_y = [1, 300];
cropped_img_size = [1400, 700]; % width, height

no_of_crops = 1;
train_or_test_bool = 2;
train_or_test = {'train', 'test'};

src_image_dir = ['/Users/reza/Class_and_research/IU_research/pytorch/electro_migration/data/selected_images_' train_or_test{train_or_test_bool} '/'];
src_images = dir(sprintf('%s/*.png',src_image_dir));
gt_dir = ['/Users/reza/Class_and_research/IU_research/pytorch/electro_migration/data/annotated_selected_images_' train_or_test{train_or_test_bool} '/'];

vis = 0;

dest_dir_img = ['/Users/reza/Class_and_research/IU_research/pytorch/electro_migration/data/cropped_image/'];
dest_dir_gt = '/Users/reza/Class_and_research/IU_research/pytorch/electro_migration/data/gt/';



for i_src_images=1:length(src_images)
    src_img_name = src_images(i_src_images).name(1:end-4);
    
    
    % crop each IC from the IC-board
    %ic_board_name_wo_suffix = ic_board_name(1:4);
    src_img = imread(sprintf('%s/%s.png', src_image_dir, src_img_name));
    [r, c, ~] = size(src_img);
%     src_gt_mask = zeros(r,c);
%     src_gt_mask(gt_coord_y(1):gt_coord_y(2), gt_coord_x(1):gt_coord_x(2)) = 1;
    
    if (train_or_test_bool == 1)
        gt = load(sprintf('%s/%s_labels.mat', gt_dir, src_img_name));
        gt = gt.objects.labels;
        gt = gt-1;
    else
        gt = zeros(r,c);
    end
   
    for i_crops=1:no_of_crops
        
        dest_img_name = sprintf('%s-%02d',src_img_name, i_crops);
        rand_x = round(init_coord_x(2)*rand());
        rand_y = round(init_coord_y(2)*rand());
        cropped_img = src_img(rand_y:rand_y+cropped_img_size(2), rand_x:rand_x+cropped_img_size(1),:);
        
        cropped_mask = gt(rand_y:rand_y+cropped_img_size(2), rand_x:rand_x+cropped_img_size(1));
        
        if (vis)
            figure; imagesc(cropped_img);
            figure; imagesc(cropped_mask);
            pause(1);
            close all;
        end
        %keyboard;
        imwrite(uint8(cropped_mask), sprintf('%s/%s.png',dest_dir_gt,dest_img_name));
        imwrite(cropped_img, sprintf('%s/%s.png',dest_dir_img,dest_img_name));
        
       
    end
    
    disp(['done cropping image ...' src_img_name]);
    %keyboard;

end