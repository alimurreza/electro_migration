     % Electro_migration project: this script renames the files and prepares images for two different
    % modalities: a) CCD image and b) Thermal image (used later for annotation/training+testing machine learning model)

    % Md Alimoor Reza
    % mdreza@iu.edu
    % Postdoctoral Associate, 
    % Indiana University Bloomington
    % January 2019


    close all; clear; clc;

    %-----------------------------------------------------------------------------------------------------------------------------------------------------
    % dataset_name_dest = 'electro_migra_001';
    % dataset_name_src = {'electro_migra_001_tmp/part1/', 'electro_migra_001_tmp/part2/'};

    % /Users/reza/Class_and_research/IU_research/thermal_image_project/thermal_image/updated_thermal_image/electro_migra_001_tmp
    % REZA: purdue people updated the mat files early January, 2019. only the
    % thermal images were modified (they told us that these new thermal images are improved version than the earlier ones)
    % dataset_name_dest = 'updated_thermal_image/electro_migra_001';
    % dataset_name_src = {'updated_thermal_image/electro_migra_001_tmp/part1/', 'updated_thermal_image/electro_migra_001_tmp/part2/'};
    %-----------------------------------------------------------------------------------------------------------------------------------------------------

    % REZA: CHANGE THE SRC_DIR once new data arrived (03/19)
    seq_name            = 8;
    %root_dir            = '/Users/mchivuku/Documents/research-projects/thermal-images/electro_migration_IU_PURDUE/';
    root_dir            = '/Users/reza/Class_and_research/IU_research/thermal_image_project/thermal_image/';
        
    is_thermal          = 1;
    is_ccdImage         = 1;
    vis                 = 0;
    is_write            = 1;
    
    if (seq_name >=1 && seq_name <= 3)
        thermal_min_val     = 0;
        thermal_max_val     = 130;
    elseif( seq_name > 4)
        thermal_min_val     = 65;
        thermal_max_val     = 155;
    else
        disp('Reza: (06/13/19) Assign the correct thermal_min and thermal_max values');
        keyboard;
    end
    
    
    %-------------------------------------------------------------------------   
    %------------        failure at the Negative pad        ------------------
    % ------------------------------------------------------------------------
        
    if (seq_name == 1)
        dataset_newname     = 'electro_migra_001';
        dataset_name_dest   = ['2018_11_12_6B_64C_100X/' dataset_newname];
        dataset_name_src    = {'2018_11_12_6B_64C_100X/electro_migra_001_tmp/part1/', '2018_11_12_6B_64C_100X/electro_migra_001_tmp/part2/'};
        total_images        = [11, 122];
        img_num_offset      = [0, 11];
        prefix              = {'2018_11_12_521_6B_19.4V_142ohms_64C_100X_1x_780nm2Cooling', '2018_11_12_521_6B_19.4V_142ohms_64C_S2_100X_1x_780nm2Cooling'};
  
    elseif(seq_name == 2)
        dataset_newname     = 'electro_migra_002';
        dataset_name_dest   = ['2018_11_19_7B_55C_100x/' dataset_newname];
        dataset_name_src    = {'2018_11_19_7B_55C_100x/'};
        total_images        = 206;
        img_num_offset      = 0;
        prefix              = {'2018_11_19_521_FWD_7B_19.2V_146ohms_55C_100X_1x_530nm2Cooling'};    

    elseif(seq_name == 3)
        dataset_newname     = 'electro_migra_003';
        dataset_name_dest   = ['2018_11_19_7T_33C_100x/' dataset_newname];
        dataset_name_src    = {'2018_11_19_7T_33C_100x/'};
        total_images        = 420;
        img_num_offset      = 0;
        prefix              = {'2018_11_19_521_FWD_7T_18.6V_147ohms_34C_85mA_100X_1x_780nm2Cooling'};    


    % ------------------------------------------------------------------------------
    % ------------------        failure at both Negative Pad and Void   ------------
    % ------------------                TO BE ANALYZED                  ------------
    
    elseif(seq_name == 4)
        dataset_newname     = 'electro_migra_004';
        dataset_name_dest   = ['2018_11_06_5B_21.5C_50x/' dataset_newname];
        dataset_name_src    = {'2018_11_06_5B_21.5C_50x/all_files/'};
        total_images        = 715;
        img_num_offset      = 0;
        prefix              = {'2018_11_06_521_5B_19V_158ohms_21C_050X_1x_530nm2Cooling'};  
    
    %--------------------------------------------------------------------------------     
    %---------------              failure at Void           -------------------------
    % ------------------------------------------------------------------------------
    
    elseif(seq_name == 5)
        dataset_newname     = 'electro_migra_005';
        dataset_name_dest   = ['8B/' dataset_newname];
        dataset_name_src    = {'8B/'};
        total_images        = 11;
        img_num_offset      = 0;
        prefix              = {'2018_12_27_521_FWD_8B_19.2V_147ohms_21C_100X_1x_780nm2Cooling'};  
    
    elseif(seq_name == 6)
        dataset_newname     = 'electro_migra_006';
        dataset_name_dest   = ['10T/' dataset_newname];
        dataset_name_src    = {'10T/'};
        total_images        = 12;
        img_num_offset      = 0;
        prefix              = {'2019_01_01_521_FWD_10T_22V_149ohms_27C_100X_1x_780nm2Cooling'};  
        
    elseif(seq_name == 7)
        dataset_newname     = 'electro_migra_007';
        dataset_name_dest   = ['11T/' dataset_newname];
        dataset_name_src    = {'11T/'};
        total_images        = 23;
        img_num_offset      = 0;
        prefix              = {'2019_01_02_521_FWD_11T_23V_149ohms_35C_100X_1x_780nm2Cooling'};  
  
    elseif(seq_name == 8)
        dataset_newname     = 'electro_migra_008';
        dataset_name_dest   = ['12B/' dataset_newname];
        dataset_name_src    = {'12B/'};
        total_images        = 11;
        img_num_offset      = 0;
        prefix              = {'2019_01_04_521_FWD_12B_151ohms_27C_100X_1x_780nm2Cooling'};  

    elseif(seq_name == 9)
        dataset_newname     = 'electro_migra_009';
        dataset_name_dest   = ['12T/' dataset_newname];
        dataset_name_src    = {'12T/'};
        total_images        = 9;
        img_num_offset      = 0;
        prefix              = {'2019_01_04_521_FWD_12T_151ohms_35C_100X_1x_780nm2Cooling'};  


    elseif(seq_name == 10)
        dataset_newname     = 'electro_migra_010';
        dataset_name_dest   = ['13B/' dataset_newname];
        dataset_name_src    = {'13B/'};
        total_images        = 7;
        img_num_offset      = 0;
        prefix              = {'2019_01_04_521_FWD_13B_146ohms_22C_100X_1x_780nm2Cooling'};  


    elseif(seq_name == 11)
        dataset_newname     = 'electro_migra_011';
        dataset_name_dest   = ['13T/' dataset_newname];
        dataset_name_src    = {'13T/'};
        total_images        = 11;
        img_num_offset      = 0;
        prefix              = {'2019_01_04_521_FWD_13T_146ohms_42C_100X_1x_780nm2Cooling'};  

    elseif(seq_name == 12)
        dataset_newname     = 'electro_migra_012';
        dataset_name_dest   = ['14T/' dataset_newname];
        dataset_name_src    = {'14T/'};
        total_images        = 7;
        img_num_offset      = 0;
        prefix              = {'2019_01_07_521_FWD_14T_153ohms_22C_100X_1x_780nm2Cooling'};  

    elseif(seq_name == 13)
        dataset_newname     = 'electro_migra_013';
        dataset_name_dest   = ['15B/' dataset_newname];
        dataset_name_src    = {'15B/'};
        total_images        = 10;
        img_num_offset      = 0;
        prefix              = {'2019_01_07_521_FWD_15B_144ohms_22C_100X_1x_780nm2Cooling'};  
        
        
    elseif(seq_name == 14)
        dataset_newname     = 'electro_migra_014';
        dataset_name_dest   = ['15T/' dataset_newname];
        dataset_name_src    = {'15T/'};
        total_images        = 12;
        img_num_offset      = 0;
        prefix              = {'2019_01_07_521_FWD_15T_144ohms_22C_100X_1x_780nm2Cooling'};  
        
        
    elseif(seq_name == 15)
        dataset_newname     = 'electro_migra_015';
        dataset_name_dest   = ['16T/' dataset_newname];
        dataset_name_src    = {'16T/'};
        total_images        = 6;
        img_num_offset      = 0;
        prefix              = {'2019_01_22_521_FWD_16T_135ohms_27C_100X_1x_780nm2Cooling'};  
        
        
    elseif(seq_name == 16)
        dataset_newname     = 'electro_migra_016';
        dataset_name_dest   = ['17T/' dataset_newname];
        dataset_name_src    = {'17T/'};
        total_images        = 8;
        img_num_offset      = 0;
        prefix              = {'2019_01_22_521_FWD_17T_27C_100X_1x_780nm2Cooling'};  
        
    elseif(seq_name == 17)
        dataset_newname     = 'electro_migra_017';
        dataset_name_dest   = ['18T/' dataset_newname];
        dataset_name_src    = {'18T/'};
        total_images        = 13;
        img_num_offset      = 0;
        prefix              = {'2019_01_22_521_FWD_18T_27C_100X_1x_780nm2Cooling'};  

    elseif(seq_name == 18)
        dataset_newname     = 'electro_migra_018';
        dataset_name_dest   = ['19B/' dataset_newname];
        dataset_name_src    = {'19B/'};
        total_images        = 6;
        img_num_offset      = 0;
        prefix              = {'2019_01_22_521_FWD_19B_35C_100X_1x_780nm2Cooling'};  

    elseif(seq_name == 19)
        dataset_newname     = 'electro_migra_019';
        dataset_name_dest   = ['19T/' dataset_newname];
        dataset_name_src    = {'19T/'};
        total_images        = 9;
        img_num_offset      = 0;
        prefix              = {'2019_01_22_521_FWD_19T_35C_100X_1x_780nm2Cooling'};  

    elseif(seq_name == 20)
        dataset_newname     = 'electro_migra_020';
        dataset_name_dest   = ['20B/' dataset_newname];
        dataset_name_src    = {'20B/'};
        total_images        = 16;
        img_num_offset      = 0;
        prefix              = {'2019_01_24_521_FWD_20B_43C_100X_1x_780nm2Cooling'};  

    elseif(seq_name == 21)
        dataset_newname     = 'electro_migra_021';
        dataset_name_dest   = ['20T/' dataset_newname];
%         dataset_name_src    = {'20T/'};
%         total_images        = 19;
%         img_num_offset      = 0;
%         prefix              = {'2019_01_24_521_FWD_20T_43C_2_100X_1x_780nm2Cooling'};  

        dataset_name_src    = {'20T/part1/', '20T/part2/'};
        total_images        = [11, 8];
        img_num_offset      = [0, 11];
        prefix              = {'2019_01_24_521_FWD_20T_43C_100X_1x_780nm2Cooling', '2019_01_24_521_FWD_20T_43C_2_100X_1x_780nm2Cooling'};

    elseif(seq_name == 22)
        dataset_newname     = 'electro_migra_022';
        dataset_name_dest   = ['21B/' dataset_newname];
        dataset_name_src    = {'21B/'};
        total_images        = 5;
        img_num_offset      = 0;
        prefix              = {'2019_01_24_521_FWD_21B_44C_100X_1x_780nm2Cooling'};  
 
    elseif(seq_name == 23)
        dataset_newname     = 'electro_migra_023';
        dataset_name_dest   = ['22B/' dataset_newname];
        dataset_name_src    = {'22B/'};
        total_images        = 19;
        img_num_offset      = 0;
        prefix              = {'2019_01_24_521_FWD_22B_100X_1x_780nm2Cooling'};  
 
    elseif(seq_name == 24)
        dataset_newname     = 'electro_migra_024';
        dataset_name_dest   = ['22T/' dataset_newname];
        dataset_name_src    = {'22T/'};
        total_images        = 26;
        img_num_offset      = 0;
        prefix              = {'2019_01_24_521_FWD_22T_55C_100X_1x_780nm2Cooling'};  
 

    elseif(seq_name == 25)
        dataset_newname     = 'electro_migra_025';
        dataset_name_dest   = ['23B/' dataset_newname];
        dataset_name_src    = {'23B/'};
        total_images        = 16;
        img_num_offset      = 0;
        prefix              = {'2019_01_25_521_FWD_23B_100X_1x_780nm2Cooling'};  

   %--------------------------------------------------------------------------------     
    else

    end


    dest_img_path       = [root_dir '/' dataset_name_dest '/'];
    %postfix = {'', ''};

    if (~exist([dest_img_path '/CCDImage'], 'dir'))
        disp(['creating new destination directory: ' dest_img_path '/CCDImage']);
        %keyboard;
        mkdir([dest_img_path '/CCDImage']);
    end

    if (~exist([dest_img_path '/Thermal'], 'dir'))
        disp(['creating new destination directory: ' dest_img_path '/Thermal']);
        %keyboard;
        mkdir([dest_img_path '/Thermal']);
    end

    for j=1:length(dataset_name_src)

        src_img_path = [root_dir '/' dataset_name_src{j} '/'];

        last_frames_correct_name = img_num_offset(j);

        if (vis)
            figure;
        end

        for i=1:total_images(j)
%         for i=total_images(j):total_images(j)

            cur_name = sprintf('%s%3d', prefix{j},i);
            cur_name = dir([src_img_path '/'  cur_name '*']);
            
    %         fprintf('%d) found %s \n', i, cur_name);
            if numel(cur_name)==0
                continue
            end
            
            cur_name = cur_name(1).name;
            
            fprintf('%d) found \n', i);

%             if (i == 11)
%                 keyboard;
%             end
            load([src_img_path '/' cur_name], 'CCDImage', 'Mask', 'Thermal'); % loads three different images
            
            % REZA (06/13/19): Pardon Reza for this hack. He is afraid that the data received from Purdue 
            % has a very unconventional way of naming them. He was trying to conform to that: 
            % a) Subset of files were saved in one directory. The remaining followed from the next directory
            %    eventhough the starting offset again starts from 1.
            % b) The last file saved in an experiment should be treated as the first file. eg, 122 is actually 
            %    file number 0.
            
            %%--------------------------------------------------------------------------------------------
            % Needs correction: see the next version
%             if (i ~= total_images(j))
%                 img_new_name = sprintf('%s_%05d.png', dataset_newname,(i+img_num_offset(j)));
%             else
%                 img_new_name = sprintf('%s_%05d.png', dataset_newname,(0+img_num_offset(j)));
%             end
            %%--------------------------------------------------------------------------------------------
            
            if (i == total_images(j)) 
                img_new_name = sprintf('%s_%05d.png', dataset_newname, last_frames_correct_name); % Two directories each containing 10 files each, then first directory's last file name is 0 (remaining are 1-9), second directory's last file name is 10 (remaining are 11-19)
            else
                img_new_name = sprintf('%s_%05d.png', dataset_newname,(i+img_num_offset(j)));                
            end


            %% covert to range [0,255]
            % images are saved into floating point value with max value 4x10^9
            max_val = max(CCDImage(:));
            min_val = min(CCDImage(:));
            CCDImage_ = 255.0*(max_val - CCDImage)/(max_val-min_val);
            CCDImage_ = 255.0 - CCDImage_; % invert the BG and FG values. Higher=FG, Lower=BG

            disp(['min: ' num2str(min(Thermal(:))) ', max: ' num2str(max(Thermal(:)))]);            
            %% Normalize the thermal images according to given Min/Max value provided by Purdue University (contact person Sami Aljouni)
            
            
%             % Manju: 06/07/19
%             for t  = 1:length(Thermal)
%                 if Thermal(t)>thermal_max_val || Thermal(t)<thermal_min_val
%                     Thermal(t) = 0;
%                 end
%             end
%             
%             ThermalImage_ = 255.0*(thermal_max_val - Thermal)/(thermal_max_val-thermal_min_val);
%             ThermalImage_ = 255.0 - ThermalImage_; % invert the BG and FG values. Higher=FG, Lower=BG

            %% Reza: 06/13/19            
            idx = find(Thermal > thermal_max_val);
            Thermal(idx)    = thermal_min_val;         % anything beyond thermal_max_value is floored to value of 0 (ceiling to thermal_max_value might negatively affect)
            idx = find(Thermal <= thermal_min_val);
            Thermal(idx)    = thermal_min_val;         % anything less than thermal_min_value is also floored to value of 0
            
            ThermalImage_   = 255.0*(thermal_max_val - Thermal)/(thermal_max_val-thermal_min_val);
            ThermalImage_   = 255.0 - ThermalImage_; % invert the BG and FG values. Higher=FG, Lower=BG

            disp(['Normalized thermal: min: ' num2str(min(ThermalImage_(:))) ', max: ' num2str(max(ThermalImage_(:)))]);
            if (vis)
%                 subplot(2,1,1); imagesc(CCDImage); title(['CCDImage: ' img_new_name])
%                 subplot(2,1,2); imagesc(Thermal); title('Thermal')
%                 
                %subplot(3,1,2); imagesc(Mask); title('Mask') % REZA: MASK
                %IMAGE IS EMPTY (THIS MASK REFERS TO A DIFFERENT PARAMETER PURDUR PEOPLE USED IN THEIR DATASET)
                
                %subplot(2,1,1); imagesc(CCDImage_); title(['CCDImage: ' img_new_name])
                subplot(2,1,1); imagesc(ThermalImage_); title('Thermal')
  
                pause(1);
    %             pause;

                clf;
            end

            if (is_write)
                fprintf('saving %s -> %s \n', cur_name,img_new_name);
                %keyboard;
                if (is_ccdImage)
                    imwrite(uint8(CCDImage_), [dest_img_path '/CCDImage/' img_new_name]);
                end
                
                if (is_thermal)
                    imwrite(uint8(ThermalImage_), [dest_img_path '/Thermal/' img_new_name]);
                end
                
            end

            clear CCDImage Mask Thermal;
            
        end

    end