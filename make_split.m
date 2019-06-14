function make_split(image_indices, files, src_dir, dest_dir, img_src_extension, img_dest_extension, split, fnToftMap)
    % training images
    file_names    = {};
    failure_times = [];
    for ii=1:length(image_indices)

        iFiles      = image_indices(ii);
        img_name    = files(iFiles).name(1:end-4);
        img_ccd     = imread(sprintf('%s/CCDImage/%s.%s', src_dir, img_name, img_src_extension));
        img_thermal = imread(sprintf('%s/Thermal/%s.%s', src_dir, img_name, img_src_extension));
        
        if (~exist(sprintf('%s/%s/uncropped_ccd',dest_dir, split), 'dir'))
            mkdir(sprintf('%s/%s/uncropped_ccd',dest_dir, split));
        end
        
        if (~exist(sprintf('%s/%s/uncropped_thermal',dest_dir, split), 'dir'))
            mkdir(sprintf('%s/%s/uncropped_thermal',dest_dir, split));
        end

        %figure; imagesc(img_ccd);
        % the structure of the path 'dest_root/train/uncropped_ccd/filename.extension' or 'dest_root/test/uncropped_thermal/filename.extension'
        imwrite(img_ccd, sprintf('%s/%s/uncropped_ccd/%s-01.%s',dest_dir, split, img_name,img_dest_extension));
        imwrite(img_thermal, sprintf('%s/%s/uncropped_thermal/%s-01.%s',dest_dir, split, img_name,img_dest_extension));
        fprintf('%d) done processing %s ...%s\n',ii,split, img_name);
        
        % failure times
        file_names{end+1} = [img_name  '-01.png']; % save with '-01' postfix (consistency with previous naming convention)
        failure_times = [failure_times; fnToftMap([img_name  '.png'])];

    end
    
    save([dest_dir '/failure_times_' split '_unsorted.mat'], 'failure_times', 'file_names');

end