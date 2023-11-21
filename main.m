clear all;clc;
% This code adapts a pre-trained model's weights for each participant
% in the benchmark [2]/BETA [3] dataset and tests the adapted network performance. 
% For details, please look at our paper [1].

% Pre-trained models are generated in the "pre_training.m" file.

% [1] Osman Berke Guney, Deniz Kucukahmetler, and Huseyin Ozkan,
% "Source Free Domain Adaptation of a DNN for SSVEP-based Brain-Computer Interfaces,”
% arXiv, 2023.

%% Preliminaries
% Please download benchmark [2] and/or BETA [3] datasets
% and add folder that contains downloaded files to the MATLAB path.

% [2] Y. Wang, X. Chen, X. Gao, and S. Gao, “A benchmark dataset for
% ssvep-based brain–computer interfaces,” IEEE Transactions on Neural Systems and
% Rehabilitation Engineering,vol. 25, no. 10, pp. 1746–1752, 2016.

% [3] B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, “Beta: A large
% benchmark database toward ssvep-bci application,” Frontiers in
% Neuroscience, vol. 14, p. 627, 2020.
%% Specifications (e.g. number of character) of datasets
subban_no=3; % # of subbands/bandpass filters
dataset='Bench'; % 'Bench' or 'BETA' dataset
signal_length=0.8; 
model_name = "main_net_" + dataset + "_" + num2str(signal_length) + "_";
if strcmp(dataset,'Bench')
    totalparticipants=35; % # of subjects
    totalblock=6; % # of blocks
    totalcharacter=40; % # of characters
    sampling_rate=250; % Sampling rate
    visual_latency=0.14; % Average visual latency of subjects
    visual_cue=0.5; % Length of visual cue used at collection of the dataset
    sample_length=sampling_rate*signal_length; % Sample length
    total_ch=64; % # of channels used at collection of the dataset
    max_epochs=1000; % # of epochs for first stage
    dropout_second_stage=0.6; % Dropout probabilities of first two dropout layers at second stage
    char_freqs = [8,9,10,11,12,13,14,15,8.20000000000000,9.20000000000000,10.2000000000000,11.2000000000000,12.2000000000000,13.2000000000000,14.2000000000000,15.2000000000000,8.40000000000000,9.40000000000000,10.4000000000000,11.4000000000000,12.4000000000000,13.4000000000000,14.4000000000000,15.4000000000000,8.60000000000000,9.60000000000000,10.6000000000000,11.6000000000000,12.6000000000000,13.6000000000000,14.6000000000000,15.6000000000000,8.80000000000000,9.80000000000000,10.8000000000000,11.8000000000000,12.8000000000000,13.8000000000000,14.8000000000000,15.8000000000000];
    addpath('D:\GitHub\SSVEP Datasets\Benchmark\SSVEP_Benchmark'); %Add a path of folder that contains the benchmark dataset
    
elseif strcmp(dataset,'BETA')
    totalparticipants=70;
    totalblock=4;
    totalcharacter=40;
    sampling_rate=250;
    visual_latency=0.13;
    visual_cue=0.5;
    sample_length=sampling_rate*signal_length; %
    total_ch=64;
    max_epochs=800;
    dropout_second_stage=0.7;    
    char_freqs=[8.60000000000000,8.80000000000000,9,9.20000000000000,9.40000000000000,9.60000000000000,9.80000000000000,10,10.2000000000000,10.4000000000000,10.6000000000000,10.8000000000000,11,11.2000000000000,11.4000000000000,11.6000000000000,11.8000000000000,12,12.2000000000000,12.4000000000000,12.6000000000000,12.8000000000000,13,13.2000000000000,13.4000000000000,13.6000000000000,13.8000000000000,14,14.2000000000000,14.4000000000000,14.6000000000000,14.8000000000000,15,15.2000000000000,15.4000000000000,15.6000000000000,15.8000000000000,8,8.20000000000000,8.40000000000000];
    addpath('D:\GitHub\SSVEP Datasets\Benchmark\SSVEP_BETA'); % Add a path of folder that contains the BETA dataset
    
    %else %if you want to use another dataset please specify parameters of the dataset
    % totalsubject= ... ,
    % totalblock= ... ,
    % ...
end

%% Preprocessing
total_delay=visual_latency+visual_cue; % Total undesired signal length in seconds
delay_sample_point=round(total_delay*sampling_rate); % # of data points correspond for undesired signal length
sample_interval = (delay_sample_point+1):delay_sample_point+sample_length; % Extract desired signal
channels=[48 54 55 56 57 58 61 62 63];% Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
% To use all the channels set channels to 1:total_ch=64;
[AllData,y_AllData]=PreProcessUpt(channels,sample_length,sample_interval,subban_no,totalparticipants,totalblock,totalcharacter,sampling_rate,dataset,1,2);
% Dimension of AllData:
sizes=size(AllData);
% (# of channels, # sample length, #subbands, # of characters, # of blocks, # of participants)

total_ins=totalcharacter*totalblock; % Total number of samples per participant
%% Load FBCCA predictions
% We get initial predictions from either the pre-trained initial model or the FBCCA method. 
% We choose the one having an initial higher silhouette score 
% (see Section D, Initial Predictions, in the APPENDIX).

% Firstly you need to run the code of "fbcca_classification.m" to generate
% predictions of FBCCA method.
fbcca_sv_name=['FBCCA_predictions_',dataset,'.mat'];
load(fbcca_sv_name);
all_signal_length = 0.2:0.2:1.0;
time_idx = (signal_length==all_signal_length);
%% Evaluations
% Define global variables that are used in the custom classification layer
% named "ourClassificationLayer.m", where we calculate our total loss
% function and MATLAB automatically differentiates the loss at the
% backpropagation stage.
global lambda;
global gr_labels;
global lambda_i;
global ind_valsx;

ind_valsx=ones(total_ins,1); % Initialize indexes of instances having positive 
% silhouette scores; those instances are used in our loss function.

threshold = 0.05; % Threshold is \delta in the paper.
% Set the threshold that determines how much change (ratio-wise) is 
% considered the significant drop in the neighborhood selection.

all_lambdas=0.0:0.2:1.0; % Define all the candidate lambda values.

% For each lambda in the all_lambdas vector, we adapt the pre-train DNN
% and store the accuracy and clustering performance of the adapted network
% in the "accuracies_lambdas" and "clustering_lambdas" matrices. At the
% final accuracy calculation stage, we choose the accuracy of adapted network  
% having a higher overall silhouette score.

% Initialization of "accuracies_lambdas" and "clustering_lambdas" matrices. 
accuracies_lambdas=zeros(totalparticipants,length(all_lambdas));
clustering_lambdas=zeros(totalparticipants,length(all_lambdas));

% For each participant, load the corresponding pre-trained DNN and adapt it:
for test_participant=1:totalparticipants
    for lambda_idx=1:length(all_lambdas)%start_idxi
        lambda=all_lambdas(lambda_idx);

        % Load global model and get test participant instances.     
        model = model_name + num2str(test_participant)+".mat";
        main_net = load(model);
        main_net = main_net.main_net;
        
        testdata=AllData(:,:,:,:,:,test_participant);
        testdata=reshape(testdata,[sizes(1),sizes(2),sizes(3),totalcharacter*totalblock]);
        
        % Define the model and transfer the weights from the pre-trained model.         
        % Set the last layer of the model to our custom layer "ourClassificationLayer",
        % where we implement our loss function.
        layers = [ ...
            imageInputLayer([sizes(1),sizes(2),sizes(3)],'Normalization','none')
            convolution2dLayer([1,1],1)
            convolution2dLayer([sizes(1),1],120)
            dropoutLayer(dropout_second_stage)
            convolution2dLayer([1,2],120,'Stride',[1,2])
            dropoutLayer(dropout_second_stage)
            reluLayer
            convolution2dLayer([1,10],120,'Padding','Same')
            dropoutLayer(0.95)
            fullyConnectedLayer(totalcharacter)
            softmaxLayer
            ourClassificationLayer];
        
        % Transfer the weights.
        layers(2, 1).Weights = main_net.Layers(2, 1).Weights;
        layers(3, 1).Weights = main_net.Layers(3, 1).Weights;
        layers(5, 1).Weights = main_net.Layers(5, 1).Weights;
        layers(8, 1).Weights = main_net.Layers(8, 1).Weights;
        layers(10, 1).Weights = main_net.Layers(10, 1).Weights;
        
        layers(2, 1).BiasLearnRateFactor=0;
        layers(3, 1).Bias = main_net.Layers(3, 1).Bias;
        layers(5, 1).Bias = main_net.Layers(5, 1).Bias;
        layers(8, 1).Bias = main_net.Layers(8, 1).Bias;
        layers(10, 1).Bias = main_net.Layers(10, 1).Bias;
        
        
        % Get the predictions of the pre-trained model and calculate the initial
        % overall silhouette score. Also, determine the neighbors of each data point.
        [YPred,~] = classify(main_net,testdata);        
        clstr_score = neighbors_calculation(main_net,testdata,YPred,sizes,threshold,total_ins);      
        
        % Get the predictions of the FBCCA method and calculate its overall silhouette score.     
        [~,corr_cca,~,~] = silhouette_score_calculation(main_net,testdata,categorical(fbcca_predictions(:,test_participant,time_idx)));
       
        % If the FBCCA method has a higher silhouette score, start the
        % adaptation with the predictions from the FBCCA method.
        if corr_cca>clstr_score
           cca_enter(test_participant)=1;            
           YPred=categorical(fbcca_predictions(:,test_participant,time_idx));
           clstr_score = neighbors_calculation(main_net,testdata,categorical(fbcca_predictions(:,test_participant,time_idx)),sizes,threshold,total_ins);
        end
        
        % Sometimes, the predictions do not contain all of the characters (classes),
        % at those times, MATLAB gives error. To avoid from this error,
        % the dumb instances are created and added to the samples of the
        % test participant (note that those instances do not have any effect).
        tmp_ind_valsx=zeros(total_ins+40,1,'logical');
        tmp_gr_labels=cell(total_ins+40,1);
        tmp_lambda_i=cell(total_ins+40,1);       
        if length(unique(YPred))~=40
           tmp_lambda_i(41:total_ins+40)=lambda_i;
           tmp_ind_valsx(41:total_ins+40)=ind_valsx;
           tmp_gr_labels(41:total_ins+40)=gr_labels;
           ind_valsx=tmp_ind_valsx;
           gr_labels=tmp_gr_labels;
           lambda_i=tmp_lambda_i;
           testdata_train=zeros(sizes(1),sizes(2),sizes(3),40);
           YPred_train=categorical((1:40)');
           testdata_train(:,:,:,end+1:end+total_ins)=testdata;
           YPred_train(end+1:end+total_ins)=YPred;
           for tmp_i=41:total_ins+40
               tmp_gr=gr_labels{tmp_i};
               if ~isempty(tmp_gr)
                   tmp_gr=tmp_gr+40;
                   gr_labels{tmp_i}= tmp_gr;
               end
           end
        else
            testdata_train=testdata;
            YPred_train=YPred;
        end        
        
        % Do adaptation
        continue_cond=1; % Define a flag variable.
        number_try_total=3; % Set maximum number of trials.
        number_try=0; % Initialize the variable holding the trial number.
        iter=1; % Initiliaze the iteration number
        pre_clstr_score=clstr_score; % Initialize the variable storing the silhouette score of the previous iteration.      
        
        while continue_cond
            % 
            if iter==1
                % Specify training options 
                options = trainingOptions('sgdm',... 
                'InitialLearnRate',0.0001,...
                'MaxEpochs',50,...
                'MiniBatchSize',int32(length(YPred_train)), ...
                'Shuffle','never',...                    
                'L2Regularization',0.001,...
                'ExecutionEnvironment','cpu');
                Ypred_pre=YPred;                
                net = trainNetwork(testdata_train,categorical(YPred_train),layers,options);
                [YPred,~] = classify(net,testdata);
                [~,clstr_score,~,~]=silhouette_score_calculation(net,testdata,YPred);
                
                if clstr_score>pre_clstr_score % Check whether the new cluster score is better than the previous one or not.                  
                    iter=iter+1;
                    pre_clstr_score=clstr_score;
                    number_try=0;
                    [~] = neighbors_calculation(net,testdata,YPred,sizes,threshold,total_ins);
                else
                    YPred=Ypred_pre;
                    number_try=number_try+1;
                end
            else
                options = trainingOptions('sgdm',... 
                'InitialLearnRate',0.0001,...
                'MaxEpochs',50,...
                'MiniBatchSize',int32(length(YPred)), ...
                'Shuffle','never',...                           
                'L2Regularization',0.001,...
                'ExecutionEnvironment','cpu');%'Momentum',0,...
                pre_net=net;
                Ypred_pre=YPred;                
                net = trainNetwork(testdata,categorical(YPred),net.Layers,options);
                [YPred,~] = classify(net,testdata);
                [channel_comb,clstr_score,~,~]=silhouette_score_calculation(net,testdata,YPred);
                
                if clstr_score>pre_clstr_score
                    %lambda=max(lambda-0.1,0);
                    iter=iter+1;
                    pre_clstr_score=clstr_score;
                    clstr_score = neighbors_calculation(net,testdata,YPred,sizes,threshold,total_ins);
                    number_try=0;
                else
                    YPred=Ypred_pre;
                    number_try=number_try+1;
                    net=pre_net;
                end
            end
            
            if number_try==number_try_total
                continue_cond=0;
            end
        end
        if iter ~=1
            [YPred,~] = classify(net,testdata);
            [channel_comb,clstr_score,~,~] = silhouette_score_calculation(net,testdata,YPred);
        else
            YPred=Ypred_pre;
            [channel_comb,clstr_score,~,~] = silhouette_score_calculation(main_net,testdata,YPred);            
        end

        % Get the true labels and calculate the accuracy of adapted network.
        test_y=y_AllData(:,:,:,test_participant);
        test_y=reshape(test_y,[1,totalcharacter*totalblock]);
        test_y=categorical(test_y);

        accuracies_lambdas(test_participant,lambda_idx)=mean(test_y==YPred');
        clustering_lambdas(test_participant,lambda_idx)=clstr_score;
        
        sv_name=['accuracies_lambdas_',num2str(signal_length),'_',dataset,'.mat'];
        save(sv_name,'accuracies_lambdas');
        sv_name=['clustering_lambdas',num2str(signal_length),'_',dataset,'.mat'];
        save(sv_name,'clustering_lambdas');         
    end
end
[acc_max_val,acc_max_idx]=max(accuracies_lambdas');
acc_max_val=acc_max_val';
[~,corr_max_idx]=max(clustering_lambdas');
max_corr_acc=zeros(totalparticipants,1);
for i=1:totalparticipants
    max_corr_acc(i)=accuracies_lambdas(i,corr_max_idx(i));
end
mean(max_corr_acc)