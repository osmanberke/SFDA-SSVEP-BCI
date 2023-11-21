function [channel_comb,clstr_score,silhoutte_val,ind_val]=silhouette_score_calculation(net,data,labels)
%% This function calculates the silhouette value for all of the instances
% and returns:
% the channel combination (channel_comb) that maximizes the overall silhouette score
% the current maximum overall silhouette score (clstr_score)
% the silhoeutte score for each instance (silhoutte_val)
% the indexes of instances having positive silhouette value (ind_val)
subband_weights=net.Layers(2, 1).Weights; % Get the current sub-band combination weights
ch_weights=squeeze(net.Layers(3, 1).Weights);
sizes=size(data);

%% Combine the sub-bands of instances
sub_data=zeros(sizes(1),sizes(2),sizes(4));
for i=1:sizes(4) 
    tmp_data=data(:,:,:,i);
    tmp_data=sum(subband_weights.*tmp_data,3);
    sub_data(:,:,i)=tmp_data;
end

%% Combine the channels of instances with each of the channel combinations from the channel combination layer
% and select the one maximizing the overall clustering performance
channel_combs_correlation=zeros(120,1);
silhoutte_vals = zeros(sizes(4),120);
for ch_comb_idx=1:120
    ch_comb=ch_weights(:,ch_comb_idx);
    combined_data=zeros(sizes(4),sizes(2));    
    % Combine the data across channels using the channel combination ch_comb
    % and then substact the mean to make cosine_distance(x_i,x_j) = 
    % 1-p(x_i,x_j)
    for ins_idx=1:sizes(4)
        combined_data(ins_idx,:)=ch_comb'*sub_data(:,:,ins_idx);
        combined_data(ins_idx,:)=combined_data(ins_idx,:)-mean(combined_data(ins_idx,:));
    end
    % Calculate the silhouette values using built-in function "silhouette"
    silhoutte_vals(:,ch_comb_idx)=silhouette(combined_data,double(labels),'cosine');
    channel_combs_correlation(ch_comb_idx)=sum(silhoutte_vals(:,ch_comb_idx));
end
% Select the channel maximizing the clustering performance and
% choose the silhouette scores of instances when they are combined with
% that selected channel
[clstr_score,idx]=sort(channel_combs_correlation,'descend');
silhoutte_val = silhoutte_vals(:,idx(1));
ind_val=(silhoutte_val>=0); % Indexes of the instances with positive silhouette value
channel_comb=ch_weights(:,idx(1));
clstr_score=clstr_score(1);
end