%% Clear everything 
clear all; close all; clc;

%% Dataset root folder template and suffix
dataFolderTmpl = '~/data/BC2_Sfx';
%dataFolderSfx = '96x64';
dataFolderSfx = '1072x712';


% Create imageDataset of all images in selected baseline folders
[baseSet, dataSetFolder] = createBCbaselineIDS6b(dataFolderTmpl, dataFolderSfx, @readFunctionTrainGN_n);
trainingSet = baseSet;

% Count number of the classes ('stable' - presrvation of the order - to use
% later for building confusion matrix)
labels = unique(trainingSet.Labels, 'stable');
[nClasses, ~] = size(labels);

% Print image count for each label
countEachLabel(trainingSet)

                        
%% Split Database into Training & Test Sets in the ratio 80% to 20% (usually comment out)
%[trainingSet, testSet] = splitEachLabel(baseSet, 0.4, 'randomize'); 

        
%% Load Pre-trained Network (AlexNet)
% AlexNet is a pre-trained network trained on 1000 object categories. 
%alex = alexnet;
resnet = resnet50;

%% Review Network Architecture 
lgraph_r = layerGraph(resnet);

%% Modify Pre-trained Network 
% AlexNet was trained to recognize 1000 classes, we need to modify it to
% recognize just nClasses classes. 
n_ll = 177;

lgraph = replaceLayer(lgraph_r, 'fc1000', fullyConnectedLayer(nClasses, 'Name', 'fc1000'));
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', classificationLayer('Name', 'ClassificationLayer_fc1000'));

%% Perform Transfer Learning
% For transfer learning we want to change the weights of the network ever so slightly. How
% much a network is changed during training is controlled by the learning
% rates. 
opts = trainingOptions('adam',...
                       'ExecutionEnvironment','parallel',...
                       'InitialLearnRate', 0.001,...
                       'MiniBatchSize', 64,...
                       'MaxEpochs', 10);
                                               
                      %'Plots', 'training-progress',...

%% Train the Network 
% This process usually takes about 60 minutes on a desktop GPU. 
myNet = trainNetwork(trainingSet, lgraph, opts);


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


%% Reliability training datasets
% Create imageDataset vector of images in selected makeup folders
[testRSets, testRDataSetFolders] = createBCtestIDSvect6b1(dataFolderTmpl, dataFolderSfx, @readFunctionTrainGN_n);


%% Create Matrix of Softmax Activations
[nMakeups, ~] = size(testRSets);
nImgs = 0;

for i=1:nMakeups
    [tn, ~] = size(testRSets{i}.Files);
    nImgs = nImgs + tn;
end

Act = zeros([nImgs nClasses]);
Verd = zeros(nImgs, 1);

%% Populate Matrix of Softmax Activations
nImgsCur = 1;
for i=1:nMakeups   
   
    
    %% Test Network Performance    
    predictedLabels = classify(myNet, testRSets{i}); 
    
    % Output per image scores
    predictedScores = predict(myNet, testRSets{i});
    predictedScoresSorted = sort(predictedScores, 2, 'descend');
    [nImages, ~] = size(predictedScores);
    
    [tn, ~] = size(testRSets{i}.Files);  
    Act(nImgsCur:nImgsCur + tn - 1, :) = predictedScoresSorted;
    
    for k=1:nImages
        
        correctClass = testRSets{i}.Labels(k);
        predictClass = predictedLabels(k);
        if correctClass == predictClass
            Verd(nImgsCur + k - 1) = 1;
        %else
        %    Verd(nImgsCur + k - 1) = 0;
        end
        
    end
    
    nImgsCur = nImgsCur + tn;
    
end


%% Train Supervisor model
nVerdicts = 2;
nLayer1 = nClasses;
nLayer2 = nClasses; %int16(nClasses/4);
nLayer3 = nClasses;
nLayer4 = nClasses;

sLayers = [
    featureInputLayer(nClasses)
    fullyConnectedLayer(nLayer1)
    reluLayer
    %dropoutLayer(0.5)
    fullyConnectedLayer(nLayer2)
    %tanhLayer
    reluLayer
    %dropoutLayer(0.5)
    fullyConnectedLayer(nLayer3)
    reluLayer
    %dropoutLayer(0.5)
    %fullyConnectedLayer(nLayer4)
    %reluLayer
    %dropoutLayer(0.5)
    fullyConnectedLayer(nVerdicts)
    softmaxLayer
    classificationLayer
];

sOptions = trainingOptions('adam', ...
    'ExecutionEnvironment','auto',...
    'MiniBatchSize', 64, ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',300, ...
    'Verbose',true, ...
    'Plots','training-progress');

Yt = categorical(Verd');

superNet = trainNetwork(Act, Yt, sLayers, sOptions);



%% Makeup datasets
mkDataSetFolder = strings(0);
mkLabel = strings(0);

% Create imageDataset vector of images in selected makeup folders
[testSets, testDataSetFolders] = createBCtestIDSvect6b(dataFolderTmpl, dataFolderSfx, @readFunctionTrainGN_n);


%%
[nMakeups, ~] = size(testSets);

mkTable = cell(nMakeups, nClasses+4);

%%

% Write per-image scores to a file
fd = fopen('predict_res50_6bm.txt','w');

fprintf(fd, "CorrectClass MaxScore MaxScoreClass Trusted Trustedscore FileName");
for l=1:nClasses
    fprintf(fd, " %s", myNet.Layers(n_ll).Classes(l));
end
fprintf(fd, "\n");


i = 1;
for i=1:nMakeups   
   
    
    %% Test Network Performance    
    predictedLabels = classify(myNet, testSets{i}); 
    
    % Output per image scores
    predictedScores = predict(myNet, testSets{i});
    
    % Supervisor network
    predictedScoresSorted = sort(predictedScores, 2, 'descend');
    supervisorPredictedLabels = classify(superNet, predictedScoresSorted); 
    supervisorPredictedScores = predict(superNet, predictedScoresSorted);
    
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
                maxScoreClass = myNet.Layers(n_ll).Classes(l);
            end
        end
    
        fprintf(fd, "%s %f %s %s %f %s", correctClass, maxScore, maxScoreClass, supervisorPredictedLabels(k), supervisorPredictedScores(k,2), testSets{i}.Files{k});
        for l=1:nClasses
            fprintf(fd, " %f", predictedScores(k, l));
        end
        fprintf(fd, "\n");
        
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
