%% BC2 dataset, classify, predict
% *5d: nomakeup train and makeup test

%% Clear everything 
clear all; close all; clc;

%% Dataset root folder template and suffix
dataFolderTmpl = '~/data/BC2_Sfx';
dataFolderSfx = '1072x712';


% Create imageDataset of all images in selected baseline folders
[baseSet, dataSetFolder] = createBCbaselineIDS5d2(dataFolderTmpl, dataFolderSfx, @readFunctionTrain_n);
trainingSet = baseSet;

% Count number of the classes ('stable' - presrvation of the order - to use
% later for building confusion matrix)
labels = unique(trainingSet.Labels, 'stable');
[nClasses, ~] = size(labels);

% Print image count for each label
countEachLabel(trainingSet)


% Scale all images to this dimension
imN = 277; % 277 - AlexNet
imM = imN;
nFlat1 = imN*imM;
nFlat = 3*nFlat1;

nPVar1 = 3072; %3072
nPVar = 3*nPVar1;
%nHFeat = 4096; 
%nHFeat2 = 4096;

% Data matrix initialization
[nImgs,~] = size(trainingSet.Files);
%nImgs = round(nImgs/4);
IM = zeros([nImgs nFlat1 3]);
Vn1 = zeros([nPVar1 nFlat1 3]);
IM3t = zeros([nPVar1 nImgs 3]);
ILab = trainingSet.Labels;
XXt = zeros([nPVar nImgs]);

save_net_file = 'bc2_net.mat';
save_evectn1_file = 'bc2_evectn1.mat';
save_cov_file = 'bc2_cov.mat';

i = 1;
for i=1:nImgs
    
    Img = imread(trainingSet.Files{i});
            
    fprintf(1, 'Currently training: %s Class: %s Sample: %d / %d\n', trainingSet.Files{i}, trainingSet.Labels(i), i, nImgs);
            
    % Convert to standard size, flatten to row, add to matrix of all 
    % training images. Rows as observations as needed by cov
    ImgS = imresize(Img, [imN imM]);
    ImgR = reshape(ImgS(:, :, 1), 1, []);
    IM(i, :, 1) = ImgR;
    ImgG = reshape(ImgS(:, :, 2), 1, []);
    IM(i, :, 2) = ImgG;
    ImgB = reshape(ImgS(:, :, 3), 1, []);
    IM(i, :, 3) = ImgB;
    
    %ImgV = reshape(ImgS, 1, []);
    %IM(i, :) = ImgV;
    
end

%%
i = 1;
for i=1:3
    % Find eigenvectors for covariance matrix and rotate data to
    % eigenvector basis
    %IM = gpuArray( int16(IM) );
    %IM = gpuArray( IM );
    %IM = distributed( IM );
    C = cov( IM(:, :, i) );
    %save(save_cov_file, 'C', '-v7.3');
    % [coeff, latent, explained] = pcacov(C);

    %C = gpuArray(C);
    %[V, D] = eig(C); %gpu

    %C = distributed(C);
    [V, ~] = eig(C, 'nobalance'); %distributed
    %[V, D] = eigs(C, nPVar1); %distributed
    clear( 'C' );

    VI = inv(V);
    clear( 'V' );
    Vn1(:, :, i) = VI(nFlat1-nPVar1+1:nFlat1, :);
    % Show eigensign 1
    %IM2 = VI * IM(:, :, i)';
    %ImgV2 = IM2(:, 1);
    %Img2 = reshape(ImgV2, [imN imM 3]);
    %imshow(Img2);    
    clear( 'VI' );
    IMX = IM(:, :, i)';
    IM3t(:, :, i) = Vn1(:, :, i) * IMX;
    clear( 'IMX' );
    
    %IM3(:, :, i) = IM2(nFlat1-nPVar1+1:nFlat1, :);

    % Check if latent from pcacov is the same as IM2Var, and how it's different
    % from the original image space variance
    %IM2Var = var(IM2');
    %IMVar = var(IM);
    %clear( 'IM2' );
    
end

%save(save_evectn1_file, 'Vn1', '-v7.3');
%load(save_evectn1_file, 'Vn1'); 

%% SS PCA ANN (on the best eigenvector basis components)
[nImgs,~] = size(trainingSet.Files);
ILab = trainingSet.Labels;

nHFeat = 4096; %nPVar
%nHFeat2 = nPVar1; %nPVar1
%nHFeat3 = 1024;
nClassificator = 4; %4 7 10 13
layers = [
    sequenceInputLayer(nPVar) 
    %featureInputLayer(nPVar) %, 'Normalization ', 'rescale-symmetric') -
    %alternative to sequenceInputLayer, but not supported in this version
    %fullyConnectedLayer(nHFeat)
    %reluLayer
    %dropoutLayer(0.5)
    %fullyConnectedLayer(nHFeat2)
    %reluLayer
    %tanhLayer
    %dropoutLayer(0.5)
    %fullyConnectedLayer(nHFeat3)
    %reluLayer
    %dropoutLayer(0.5)
    fullyConnectedLayer(nClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto',...
    'MiniBatchSize', 64, ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',100, ...
    'Verbose',true, ...
    'Plots','training-progress');

% Drop dimension to the best (highest variance) basis eigenvectors
for i=0:2
    XXt(i*nPVar1+1:(i+1)*nPVar1, :) = IM3t(:, :, i+1);
end
%clear( 'IM3t' );
Yt = categorical(ILab(1:nImgs)');

myNet = trainNetwork(XXt, Yt, layers, options);

%save(save_net_file, 'net', '-v7.3');
                        
%% Split Database into Training & Test Sets in the ratio 80% to 20% (usually comment out)
%[trainingSet, testSet] = splitEachLabel(baseSet, 0.4, 'randomize'); 

        
%% Load Pre-trained Network (AlexNet)
% AlexNet is a pre-trained network trained on 1000 object categories. 
%alex = alexnet;
%google = googlenet;
%vgg = vgg19;
resnet = resnet50;

%% Review Network Architecture 
%layers = alex.Layers;

%% Modify Pre-trained Network 
% AlexNet was trained to recognize 1000 classes, we need to modify it to
% recognize just nClasses classes. 
%layers(23) = fullyConnectedLayer(nClasses); % change this based on # of classes
%layers(25) = classificationLayer;

%% Perform Transfer Learning
% For transfer learning we want to change the weights of the network ever so slightly. How
% much a network is changed during training is controlled by the learning
% rates. 
%opts = trainingOptions('sgdm',...
%                       'ExecutionEnvironment','parallel',...
%                       'InitialLearnRate', 0.001,...
%                       'MiniBatchSize', 64);
                        
                      %'ExecutionEnvironment','parallel',...                          
                      %'MaxEpochs', 20,... 
                        
                      %'Plots', 'training-progress',...

%% Train the Network 
% This process usually takes about 5-20 minutes on a desktop GPU. 
%myNet = trainNetwork(trainingSet, layers, opts);


%% Traditional accuracy (usually comment out)
%predictedLabels = classify(myNet, testSet); 
%accuracy = mean(predictedLabels == testSet.Labels)

%predictedScores = predict(myNet, testSet);
%[nImages, ~] = size(predictedScores);
%for k=1:nImages
%    maxScore = 0;
%    maxScoreNum = 0;
%    maxScoreClass = "S";
%    correctClass = testSet.Labels(k);
%    for l=1:nClasses
%        if maxScore <= predictedScores(k, l)
%            maxScore = predictedScores(k, l);
%            maxScoreNum = l;
%            maxScoreClass = myNet.Layers(25).Classes(l);
%        end
%    end   
%    fprintf("%s %f %s \n", correctClass, maxScore, maxScoreClass);
%end


%% Makeup datasets
mkDataSetFolder = strings(0);
mkLabel = strings(0);

% Create imageDataset vector of images in selected makeup folders
[testSets, testDataSetFolders] = createBCtestIDSvect5d2(dataFolderTmpl, dataFolderSfx, @readFunctionTrain_n);


%
[nMakeups, ~] = size(testSets);

mkTable = cell(nMakeups, nClasses+4);

%

% Write per-image scores to a file
fd = fopen('predict_pca5d2.txt','w');


i = 1;
for i=1:nMakeups   
   
    % Data matrix initialization
    [nImgs,~] = size(testSets{i}.Files);
    IM = zeros([nImgs nFlat1]);
    IM3 = zeros([nPVar1 nImgs 3]);
    ILab = testSets{i}.Labels;
    XX = zeros([nPVar nImgs]);

    j = 1;
    for j=1:nImgs
    
        Img = imread(testSets{i}.Files{j});
            
        fprintf(1, 'Currently testing: %s Class: %s Sample: %d / %d\n', testSets{i}.Files{j}, testSets{i}.Labels(j), j, nImgs);
            
        % Convert to standard size, flatten to row, add to matrix of all 
        % training images. Rows as observations as needed by cov
        ImgS = imresize(Img, [imN imM]);
        ImgR = reshape(ImgS(:, :, 1), 1, []);
        IM(j, :, 1) = ImgR;
        ImgG = reshape(ImgS(:, :, 2), 1, []);
        IM(j, :, 2) = ImgG;
        ImgB = reshape(ImgS(:, :, 3), 1, []);
        IM(j, :, 3) = ImgB;
    
    end
    
    j = 1;
    for j=1:3
        IMX = IM(:, :, j)';
        %IM2 = Vn1(:, :, j) * IMX;
        IM3(:, :, j) = Vn1(:, :, j) * IMX;
        %IM3(:, :, j) = IM2(nFlat1-nPVar1+1:nFlat1, :);
    end
        
    % Drop dimension to the best (highest variance) basis eigenvectors
    for j=0:2
        %XX(j*nPVar1+1:(j+1)*nPVar1, :) = IM2(nFlat1-nPVar1+1:nFlat1, :);
        XX(j*nPVar1+1:(j+1)*nPVar1, :) = IM3(:, :, j+1);
    end
    Y = categorical(ILab(1:nImgs)');

    %% Test Network Performance    
    predictedLabels = classify(myNet, XX);
    predictedLabels = predictedLabels';
    
    % Output per image scores
    predictedScores = predict(myNet, XX);
    predictedScores = predictedScores';
    [nImages, ~] = size(predictedScores);
    for k=1:nImages
    
        maxScore = 0;
        maxScoreNum = 0;
        maxScoreClass = "S";
        correctClass = testSets{i}.Labels(k);
        for l=1:nClasses
            if maxScore <= predictedScores(k, l)
                maxScore = predictedScores(k, l);
                maxScoreNum = l;
                maxScoreClass = myNet.Layers(nClassificator).Classes(l);
            end
        end
    
        fprintf(fd, "%s %f %s %s\n", correctClass, maxScore, maxScoreClass, testSets{i}.Files{k});
    end
    
    [tmpStr, ~] = strsplit(testSets{i}.Files{1}, '/');
    fprintf("%s", tmpStr{1,7}); 
    mean(predictedScores)
    
    
    %% Compute average accuracy
    meanMkAcc = mean(predictedLabels == testSets{i}.Labels);
    mkTable{i,1} = testDataSetFolders(i);
    mkTable{i,2} = meanMkAcc;
    
    %%
    [tn, ~] = size(testSets{i}.Files);
    
    meanMkConf = zeros(1, nClasses);

    maxAccCat = '';
    maxAcc = 0;
    
    %%    
    %labels = string(unique(allImages.Labels, 'stable'))';
    j = 1;   
    for j = 1:nClasses

        tmpStr = strings(tn,1);
        tmpStr(:) = string(labels(j));
    
        meanMkConf(j) = mean(string(predictedLabels) == tmpStr);
        mkTable{i, 4+j} = meanMkConf(j);
        
        %find the best category match
        if maxAcc <= meanMkConf(j)
            maxAccCat = tmpStr(j);
            maxAcc = meanMkConf(j);
        end
        
    end
    mkTable{i,3} = maxAccCat;
    mkTable{i,4} = maxAcc;
    
end

%% Results
varNames = cellstr(['TestFolder' 'Accuracy' 'BestGuess' 'GuessScore' string(labels)']);
cell2table(mkTable, 'VariableNames', varNames)

fclose(fd);
