%% Author: Veronica Mandelli
% last update: 14_10_2020
% 
% use tdfReadData3D to open the file "traccati"
% in this script raw kinematic data (.tdf) will be converted into csv for
% subsequent analysis
% for each subject all the available trials will be converted into the
% .csv, a matrix will be saved to store for each subject 
%% define paths
clc, clear all
rootpath = '';
data_path = fullfile(rootpath,'data','Medea_kinematic_rawdata');

ex_file_path = fullfile(rootpath,'data','ex_subj','asd_002');
tdf_fucnt_name = fullfile(rootpath,'code','TDF');
saving_path = fullfile(rootpath,'data','all_data_csv');
code_path = fullfile(rootpath,'code');

%% load and read .tdf data
%list all the subjects in the raw data path
cd (data_path)
subject_fold = dir(data_path);
% delate first 2 arguments ({.} ,{..} and {'.DS_Store'})
subject_list = subject_fold(4:end);

% prepare the saving matrices for good or bad trial for each subject

% summary for each subject
Trials_Number = table('Size',[length(subject_list),2],'VariableTypes',{'double','double'},'VariableNames',{'Good', 'Bad'});
Trials_Number.Properties.RowNames ={subject_list().name};

% single trials for each subject
Table_trial = table('Size',[length(subject_list),17],'VariableTypes',{'string','string','string','string','string','string','string','string','string','string','string','string','string',...
    'string','string','string','string'},'VariableNames',{'trial_0', 'trial_1','trial_2','trial_3','trial_4','trial_5','trial_6','trial_7','trial_8','trial_9','trial_10',...
    'trial_11','trial_12','trial_13', 'trial_14','trial_15','subID'});

% go into the folder with the tdf functions (given by the producer of the  optoelectonic system)
cd (tdf_fucnt_name)

% loop over the subjects
for subj = 1:length(subject_list)
    
    % name of the subject
    NAME = subject_list(subj).name;
    file_path = strcat(data_path,"/",NAME,"/");
    sub(subj).id = NAME;
    
  
    % enter into subject folder to save trials names
    cd (file_path)
    trials_list = dir('*.tdf');
    disp(strcat('subject'," ", NAME," ",'has'," ", string(length(trials_list)), " ","trials"));
    
    % variables to save trials which trial is good and which is not 
    good = 0;
    bad = 0;
    Table_trial{subj,'subID'}= {NAME};
    
    % go back to functions folder
    cd (tdf_fucnt_name)
    for num = 1:length(trials_list)
        
        % load data
        trial = trials_list(num).name;
        file_name = strcat(file_path,string(trial));
        
        %save trial_order in a non-ambigous format
        trial_order_pre = regexp(trial, '\d+', 'match');
        trial_order = trial_order_pre{end};
        col = strcat('trial_',string(trial_order));
        
        % read the file with specific function
        [FREQUENCY,D,R,T,LABELS,LINKS,TRACKS] = tdfReadData3D(file_name);
        if isempty(FREQUENCY)
            disp(strcat('subject'," ", NAME," ",'TRIAL'," ", string(trial), " ","has no 3D data"));
            bad = bad+1;  
            Table_trial(subj,col)= {'bad'}; % + 1 because the 1st col is subj_ID
        else
            good =  good+1; 
            Table_trial(subj,col)= {'good'};
            % save in matrix
            % sub(subj).id = NAME;
            sub(subj).trial(num).numberGoodTrial = good;
            sub(subj).trial(num).FREQUENCY = FREQUENCY;
            sub(subj).trial(num).D = D;
            sub(subj).trial(num).R = R;
            sub(subj).trial(num).T = T;
            sub(subj).trial(num).LABELS = LABELS;
            sub(subj).trial(num).LINKS = LINKS;
            sub(subj).trial(num).fileorigin =  trial;
            
            
            % extract labels names for body parts, add X,Y,Z  to prepare
            % colNames for TRACKS1
            Subj_lables_list = [];
            LablesMat = num2cell(reshape(LABELS.',10,[]));
            [Nrow, Ncol]=size(TRACKS);
            for j  = 1: (Ncol/3)
                X_axis = "X_";
                Y_axis = "Y_";
                Z_axis = "Z_";
                for i =  1:10
                    X_axis = strcat(X_axis,LablesMat(i,j));
                    Y_axis = strcat(Y_axis,LablesMat(i,j));
                    Z_axis = strcat(Z_axis,LablesMat(i,j));
                end
                Subj_lables_list = [Subj_lables_list X_axis Y_axis Z_axis];
            end
            
            TRACKS1 = array2table(TRACKS);
            TRACKS1.Properties.VariableNames(1:Ncol) = Subj_lables_list;
            %                                                  {'45met_X','45met_Y','45met_Z', ...
            %                                                  'WristMed_X', 'WristMed_Y','WristMed_Z', ...
            %                                                  'WristLat_X', 'WristLat_Y','WristLat_Z',...
            %                                                  'Elbow_X','Elbow_Y','Elbow_Z',...
            %                                                  'Shoulder_X','Shoulder_Y','Shoulder_Z',...
            %                                                  'Castle1_X','Castle1_Y','Castle1_Z',...
            %                                                  'Castle2_X','Castle2_Y','Castle2_Z',...
            %                                                  'Castle3_X','Castle3_Y','Castle3_Z',...
            %                                                  'Castle4_X','Castle4_Y','Castle4_Z',...
            %                                                  'Ball1_X','Ball1_Y','Ball1_Z',...
            %                                                  'Ball2_X','Ball2_Y','Ball2_Z',...
            %                                                  'Ball3_X','Ball3_Y','Ball3_Z'};
            
            sub(subj).trial(num).TRACKS = TRACKS1;
            %trial_order = split(trial,".");
            %trial_order_bis = split(trial_order{2},".");
           
%             if length(trial_order_pre)==1
%                   trial_order= trial_order_pre{1};
%             elseif length(trial_order_pre)==2
%                   trial_order= trial_order_pre{2};
%             else
%                 trial_order = trial_order_pre{end};
%             end
            file_to_save = strcat(saving_path,"/",NAME,"_trial_",trial_order,'.csv');
           %comment if you do not want to save new .csv
            %writetable(TRACKS1,file_to_save,'WriteRowNames',false) 
            
        end
    
    end
    
    Trials_Number(NAME,'Good') = {good};
    Trials_Number(NAME,'Bad')= {bad};
 
end
writetable(Trials_Number,fullfile(saving_path,'/','subject_list.csv'),'WriteRowNames',true);
writetable(Table_trial,fullfile(saving_path,'/','TrialGoodBad.csv'),'WriteRowNames',true);

% end