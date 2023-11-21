clear all;clc;
% Unofficial implementation of the FBCCA method proposed in [1].

% [1] X. Chen, Y. Wang, S. Gao, T.-P. Jung, and X. Gao, “Filter bank
% canonical correlation analysis for implementing a high-speed SSVEP-
% based brain–computer interface," Journal of Neural Engineering, vol. 12,
% p. 046008, jun 2015.

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
if strcmp(dataset,'Bench')
    totalsubject=35; % # of subjects
    totalblock=6; % # of blocks
    totalcharacter=40; % # of characters
    sampling_rate=250; % Sampling rate
    visual_latency=0.14; % Average visual latency of subjects
    visual_cue=0.5; % Length of visual cue used at collection of the dataset
    total_ch=64; % # of channels used at collection of the dataset
    char_freqs = [8,9,10,11,12,13,14,15,8.20000000000000,9.20000000000000,10.2000000000000,11.2000000000000,12.2000000000000,13.2000000000000,14.2000000000000,15.2000000000000,8.40000000000000,9.40000000000000,10.4000000000000,11.4000000000000,12.4000000000000,13.4000000000000,14.4000000000000,15.4000000000000,8.60000000000000,9.60000000000000,10.6000000000000,11.6000000000000,12.6000000000000,13.6000000000000,14.6000000000000,15.6000000000000,8.80000000000000,9.80000000000000,10.8000000000000,11.8000000000000,12.8000000000000,13.8000000000000,14.8000000000000,15.8000000000000];
    addpath('D:\GitHub\SSVEP Datasets\Benchmark\SSVEP_Benchmark'); %Add a path of folder that contains the benchmark dataset

elseif strcmp(dataset,'BETA')
    totalsubject=70;
    totalblock=4;
    totalcharacter=40;
    sampling_rate=250;
    visual_latency=0.13;
    visual_cue=0.5;
    total_ch=64;
    char_freqs=[8.60000000000000,8.80000000000000,9,9.20000000000000,9.40000000000000,9.60000000000000,9.80000000000000,10,10.2000000000000,10.4000000000000,10.6000000000000,10.8000000000000,11,11.2000000000000,11.4000000000000,11.6000000000000,11.8000000000000,12,12.2000000000000,12.4000000000000,12.6000000000000,12.8000000000000,13,13.2000000000000,13.4000000000000,13.6000000000000,13.8000000000000,14,14.2000000000000,14.4000000000000,14.6000000000000,14.8000000000000,15,15.2000000000000,15.4000000000000,15.6000000000000,15.8000000000000,8,8.20000000000000,8.40000000000000];
    addpath('C:\Users\bg060\Documents\MATLAB\SSVEP\BETA Dataset'); %Add a path of folder that contains the BETA dataset

    %else %if you want to use another dataset please specify parameters of the dataset
    % totalsubject= ... ,
    % totalblock= ... ,
    % ...
end

%% Get the predictions FBCCA method for all participants
total_ins=totalcharacter*totalblock;
all_signal_length=0.2:0.2:1.0;
fbcca_predictions=zeros(total_ins,totalsubject,length(all_signal_length));
sv_name=['FBCCA_predictions_',dataset,'.mat'];


for all_signal_length_idx=1:length(all_signal_length)
    %% Data preprocessing
    signal_length = all_signal_length(all_signal_length_idx);
    sample_length= int32(sampling_rate*signal_length); % Sample length
    total_delay=visual_latency+visual_cue; % Total undesired signal length in seconds
    delay_sample_point=round(total_delay*sampling_rate); % # of data points correspond for undesired signal length
    sample_interval = (delay_sample_point+1):delay_sample_point+sample_length; % Extract desired signal
    channels=[48 54 55 56 57 58 61 62 63];% Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
    [AllDataCCA,y_AllDataCCA]=PreProcessUpt(channels,sample_length,sample_interval,5,totalsubject,totalblock,totalcharacter,sampling_rate,dataset,3,2);
    sizes=size(AllDataCCA);
    %% Define reference signals
    ref_signals=zeros(10,sample_length,totalcharacter);
    t= (0):(1/250):(signal_length-1/250);
    for i=1:totalcharacter
        tmp_ref = [ sin(2*pi*t*char_freqs(i));...
            cos(2*pi*t*char_freqs(i));...
            sin(4*pi*t*char_freqs(i));...
            cos(4*pi*t*char_freqs(i));...
            sin(6*pi*t*char_freqs(i));...
            cos(6*pi*t*char_freqs(i));...
            sin(8*pi*t*char_freqs(i));...
            cos(8*pi*t*char_freqs(i));...
            sin(10*pi*t*char_freqs(i));...
            cos(10*pi*t*char_freqs(i))];
        ref_signals(:,:,i)=tmp_ref;
    end
    a=1.25;
    b=0.25;
    weight_sub=(1:5).^(-a) + b;
    %% FBCCA Classification
    for participant=1:totalsubject
        testdata_cca=AllDataCCA(:,:,:,:,:,participant);
        testdata_cca=reshape(testdata_cca,[sizes(1),sizes(2),5,total_ins]);
        for ins=1:total_ins
            tmp_corrs=zeros(totalcharacter,1);
            %tmp_chs=zeros(9,totalcharacter,5);
            for i=1:totalcharacter
                for j=1:5
                    [~,~,rhos]=canoncorr(testdata_cca(:,:,j,ins)',ref_signals(:,:,i)');
                    tmp_corrs(i)=tmp_corrs(i)+(rhos(1)^2)*weight_sub(j);
                end
            end
            [~,fbcca_predictions(ins,participant,all_signal_length_idx)]=max(tmp_corrs);
        end
    end
end
save(sv_name,'fbcca_predictions');