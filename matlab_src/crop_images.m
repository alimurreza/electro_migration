function crop_images(dataset, split_id, global_par)

% this function crops images according to the cropping parameters (Manually adjusted for each experiment since it varies acrosse different experiment)
% images are saved according to the split (train/val/test).
% as meta data the cropping parameters are also saved for later
% introspection if required.

% Md Alimoor Reza
% mdreza@iu.edu
% Postdoctoral Associate, 
% Indiana University Bloomington
% June 2019

    split_names             = global_par.split_names;
    root_dir                = global_par.root_dir;
    split                   = split_names{split_id};
    
    if (dataset == 1)
        init_coord_x        = [700, 750];
        init_coord_y        = [400, 450];
  
    elseif(dataset == 2)
        init_coord_x        = [500, 550];
        init_coord_y        = [425, 475];
   
    elseif(dataset == 3)
        init_coord_x        = [375, 425];
        init_coord_y        = [425, 475];
    
    elseif (dataset == 5 )
        init_coord_x        = [700, 1300];
        init_coord_y        = [400, 1000];
        
    elseif(dataset == 6)
        init_coord_x        = [650, 1250];
        init_coord_y        = [150, 750];
        
    elseif(dataset == 7)
        init_coord_x        = [550, 1150];
        init_coord_y        = [100, 600];
        
    elseif(dataset == 8)
        init_coord_x        = [450, 1050];
        init_coord_y        = [250, 850];
        
    elseif(dataset == 9)
        init_coord_x        = [550, 1150];
        init_coord_y        = [250, 850];
        
    elseif(dataset == 10)
        init_coord_x        = [350, 950];
        init_coord_y        = [250, 850];
        
    elseif(dataset == 11)
        init_coord_x        = [450, 1050];
        init_coord_y        = [250, 850];
        
    elseif(dataset == 12)
        init_coord_x        = [450, 1050];
        init_coord_y        = [250, 850];
        
    elseif(dataset == 13)
        init_coord_x        = [450, 1050];
        init_coord_y        = [250, 850];
        
    elseif(dataset == 14)
        init_coord_x        = [450, 1050];
        init_coord_y        = [250, 850];
          
    elseif(dataset == 15)
        init_coord_x        = [450, 1050];
        init_coord_y        = [250, 850];
          
    elseif(dataset == 16)
        init_coord_x        = [450, 1050];
        init_coord_y        = [250, 850];
          
    elseif(dataset == 17)
        init_coord_x        = [450, 1050];
        init_coord_y        = [250, 850];

    elseif(dataset == 18)
        init_coord_x        = [450, 1050];
        init_coord_y        = [250, 850];

    elseif(dataset == 19)
        init_coord_x        = [450, 1050];
        init_coord_y        = [250, 850];

    elseif(dataset == 20)
        init_coord_x        = [450, 1050];
        init_coord_y        = [250, 850];

     elseif(dataset == 21)
        init_coord_x        = [450, 1050];
        init_coord_y        = [250, 850];
        
    elseif(dataset == 22)
        init_coord_x        = [450, 1050];
        init_coord_y        = [250, 850];
        
    elseif(dataset == 23)
        init_coord_x        = [550, 1150];
        init_coord_y        = [250, 850];
        
    elseif(dataset == 24)
        init_coord_x        = [550, 1150];
        init_coord_y        = [250, 850];
        
    elseif(dataset == 25)
        init_coord_x        = [550, 1150];
        init_coord_y        = [250, 850];
        
        
    else
        disp('NO DATASET DEFINED: ...');
        keyboard;
        
    end

    if (dataset < 4)
        cropped_img_size    = [300, 300]; % width, height
        resize_img_size     = [];
    elseif (dataset >= 5)
        cropped_img_size    = [600, 600]; % width, height (for experiment 5-25 the failure happens on disparate locations hence better to analyze more regions)
        resize_img_size     = [301, 301]; % REZA: 05/29/19: 
    end
    
    crop_params         = struct('image_name', [], 'init_x', [], 'init_y', [], 'end_x', [], 'end_y', []);
    crop_param_dir      = [root_dir '/data/electro_migra_' sprintf('%03d',dataset) '/' split '/'];
    src_ccd_dir         = [root_dir '/data/electro_migra_' sprintf('%03d',dataset) '/' split '/uncropped_ccd'];
    src_thermal_dir     = [root_dir '/data/electro_migra_' sprintf('%03d',dataset) '/' split '/uncropped_thermal'];
    dest_ccd_dir        = [root_dir '/data/electro_migra_' sprintf('%03d',dataset) '/' split '/cropped_ccd'];
    dest_thermal_dir    = [root_dir '/data/electro_migra_' sprintf('%03d',dataset) '/' split '/cropped_thermal'];
    load(                 [root_dir '/data/electro_migra_' sprintf('%03d',dataset) '/failure_times_' split '_unsorted.mat'] );    
    
    img_src_extension   = 'png';
    img_dest_extension  = 'png';
    vis = 0; if (vis) figure; end
        
    for iFiles=1:length(file_names)

        img_name    = file_names{iFiles};                                         % we are reusing the same train/test split but with a different crop from the original image.
        img_name    = img_name(1:end-4); 

        img_ccd         = imread(sprintf('%s/%s.%s', src_ccd_dir, img_name, img_src_extension));
        img_thermal     = imread(sprintf('%s/%s.%s', src_thermal_dir, img_name, img_src_extension));
        
        rand_x              = round(50*rand());
        rand_y              = round(50*rand());
        cropped_ccd         = img_ccd(init_coord_y(1)+rand_y: init_coord_y(1)+rand_y+cropped_img_size(2), init_coord_x(1)+rand_x:init_coord_x(1)+rand_x+cropped_img_size(1),:);
        cropped_thermal     = img_thermal(init_coord_y(1)+rand_y: init_coord_y(1)+rand_y+cropped_img_size(2), init_coord_x(1)+rand_x:init_coord_x(1)+rand_x+cropped_img_size(1),:);      
        
        if (~exist(sprintf('%s',dest_ccd_dir), 'dir'))
            mkdir(sprintf('%s',dest_ccd_dir));
        end
        
        if (~exist(sprintf('%s',dest_thermal_dir), 'dir'))
            mkdir(sprintf('%s',dest_thermal_dir));
        end
        
        if (vis)
            imagesc(cropped_ccd); title(['CCD: ' img_name]);
            pause;
            clf;
            imagesc(cropped_thermal); title(['Thermal:' img_name]);
            pause;
            clf;
        end
        
        % the structure of the path 'dest_root/train/bike/filename.extension' or 'dest_root/test/cat/filename.extension'
%         imwrite(cropped_ccd, sprintf('%s/%s-01.%s',dest_ccd_dir, img_name, img_dest_extension));
        %% write the CCD cropped image
        if (~isempty(resize_img_size))
            cropped_ccd = imresize(cropped_ccd, resize_img_size);
        end
        imwrite(cropped_ccd, sprintf('%s/%s.%s',dest_ccd_dir, img_name, img_dest_extension));

        %% write the Thermal cropped image
        if (~isempty(resize_img_size))
            cropped_thermal = imresize(cropped_thermal, resize_img_size);
        end
        imwrite(cropped_thermal, sprintf('%s/%s.%s',dest_thermal_dir, img_name, img_dest_extension));
        fprintf('%d) done processing %s ...%s\n',iFiles,split, img_name);
        
        % failure times
%         file_names{end+1} = [img_name  '.png'];
%         failure_times = [failure_times; fnToftMap([img_name  '.png'])];
% 

        crop_params(iFiles).image_name = img_name;
        crop_params(iFiles).init_x  = init_coord_x(1)+rand_x;
        crop_params(iFiles).init_y  = init_coord_x(1)+rand_x+cropped_img_size(1);
        crop_params(iFiles).end_x   = init_coord_y(1)+rand_y;
        crop_params(iFiles).end_y   = init_coord_y(1)+rand_y+cropped_img_size(2);
        
    end
    
    save(sprintf('%s/cropped_params.mat',crop_param_dir), 'crop_params');
%     save(['failure_times_' split '.mat'], 'failure_times', 'file_names');

end