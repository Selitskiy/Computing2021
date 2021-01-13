function [imageDS, dataSetFolders] = createBCbaselineIDS6b(dataFolderTmpl, dataFolderSfx, readFcn)

% Create a real folder
dataFolder = strrep(dataFolderTmpl, 'Sfx', dataFolderSfx);


%% Create a vector of the baseline sets

% Empty vector
dataSetFolders = strings(0);
labels = strings(0);

% Let's populate the vector by the baseline folder templates, one by one
% 
dataSetFolders = [dataSetFolders, 'S1/S1NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%dataSetFolders = [dataSetFolders, 'S1/S1NM2'];
%
dataSetFolders = [dataSetFolders, 'S2/S2NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%dataSetFolders = [dataSetFolders, 'S2/S2NM2'];
%
dataSetFolders = [dataSetFolders, 'S3/S3NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
dataSetFolders = [dataSetFolders, 'S3/S3NM2']; 
%dataSetFolders = [dataSetFolders, 'S3/S3NM3']; 
%
dataSetFolders = [dataSetFolders, 'S4/S4NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%dataSetFolders = [dataSetFolders, 'S4/S4NM2'];
%
dataSetFolders = [dataSetFolders, 'S5/S5NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
dataSetFolders = [dataSetFolders, 'S5/S5NM2'];
dataSetFolders = [dataSetFolders, 'S5/S5NM3'];
%dataSetFolders = [dataSetFolders, 'S5/S5NM4'];
%
dataSetFolders = [dataSetFolders, 'S6/S6NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%
dataSetFolders = [dataSetFolders, 'S7/S7NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
dataSetFolders = [dataSetFolders, 'S7/S7NM2'];
dataSetFolders = [dataSetFolders, 'S7/S7NM3'];
%dataSetFolders = [dataSetFolders, 'S7/S7NM4'];
%
dataSetFolders = [dataSetFolders, 'S8/S8NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%
dataSetFolders = [dataSetFolders, 'S9/S9NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%
dataSetFolders = [dataSetFolders, 'S10/S10NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%dataSetFolders = [dataSetFolders, 'S10/S10NM2'];
%
dataSetFolders = [dataSetFolders, 'S11/S11NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
dataSetFolders = [dataSetFolders, 'S11/S11NM2'];
%dataSetFolders = [dataSetFolders, 'S11/S11NM3'];
%
dataSetFolders = [dataSetFolders, 'S12/S12NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur]; 
%dataSetFolders = [dataSetFolders, 'S12/S12NM2'];
%
dataSetFolders = [dataSetFolders, 'S13/S13NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S14/S14NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%dataSetFolders = [dataSetFolders, 'S14/S14NM2'];
%
dataSetFolders = [dataSetFolders, 'S15/S15NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S16/S16NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S17/S17NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S18/S18NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S19/S19NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S20/S20NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];
%
dataSetFolders = [dataSetFolders, 'S21/S21NM1'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,1);
labels = [labels, labelCur];


%% Replace Sfx template with the actual value
dataSetFolders = strrep(dataSetFolders, 'Sfx', dataFolderSfx);
labels = strrep(labels, 'Sfx', dataFolderSfx);

% Build a full path  
fullDataSetFolders = fullfile(dataFolder, dataSetFolders);


[~, m] = size(labels);


%% Create a Datastore of the baseline images with individual folder lables
imageDS = imageDatastore(fullDataSetFolders, 'IncludeSubfolders', false,...
                            'LabelSource', 'none');                        
imageDS.ReadFcn = readFcn;


%% Label images by top subject's folder name                        
[n, ~] = size(imageDS.Files);  
labelStr = strings(n,1);

fprintf("Labeling base set...\n");

for j=1:m
    
    fprintf("Label %s\n", labels(j));
     
    matches = contains(string(imageDS.Files), labels(j));
    labelStr(matches) = labels(j); 
    
end

imageDS.Labels = categorical(labelStr);

end

