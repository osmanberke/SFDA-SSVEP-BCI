function [clstr_score] = neighbors_calculation(net,testdata,YPred,sizes,threshold,total_ins)
%% This function identifies the neighbors of each instances:

% The indexes of neighbors for each instance are stored in the global variable 
% named "gr_labels".

% The indexes of instances having positive silhouette value (ind_val) are
% assigned to the global variable of "ind_valsx". Those indexes are
% calculated in the "silhouette_score_calculation" function.

% Sometimes, one of the loss terms (or both) can be inapplicable for some instances, 
% in those cases, the lambda term should be adjusted for each instance 
% to enable the instances to contribute equally to the overall loss term.
% Those instance specific lambda values are calculated here and stored
% in the global variable named "lambda_i". 
% (see Section C, Instance Confidence, in the APPENDIX).

% This function also returns the scurrent clustering score.
nbrs_by_diff_corrs = zeros(total_ins,1);
global gr_labels;
global ind_valsx;
global lambda_i;
global lambda;
subband_weights=net.Layers(2, 1).Weights; % Get subband combination weights from network
sub_data=zeros(sizes(1),sizes(2),total_ins);
for i=1:total_ins %for subband combination
    tmp_data=testdata(:,:,:,i);
    tmp_data=sum(subband_weights.*tmp_data,3);
    sub_data(:,:,i)=tmp_data;
end

%% Calculate Nearest Neighbors of the Instances based on the Correlation Coefficient
% (see subsection of Local-Regularity Loss, in the Section IV (Method)).

[channel_comb,clstr_score,~,ind_val]=silhouette_score_calculation(net,testdata,YPred); % Get the channel combination maximizing overall clustering score
ind_valsx=ind_val; % Indexes of the instances with positive silhouette value

% For each instance, compute the correlation coefficient with every other instance
% using the channel combination "channel_comb" and sort them:
corr_ins = zeros(total_ins,total_ins);
for i = 1:total_ins
    for j=1:total_ins        
        tmp = corrcoef(sub_data(:,:,i)'*channel_comb, ...
                       sub_data(:,:,j)'*channel_comb);
        tmp = tmp(1,2);
        corr_ins(i,j) = tmp+corr_ins(i,j);        
    end
end
sorted_corrs = zeros(total_ins,total_ins);
sorted_idxs = zeros(total_ins,total_ins);

for i=1:total_ins %Sort the correlations
    [sorted_corrs(i,:),sorted_idxs(i,:)] = sort(corr_ins(i,:),'descend');
end

% For each instance, find the difference in the sorted correlation values
% calculated above to be able to detect the index, 
% where significant correlation change happens:
diffs = zeros(total_ins,total_ins-2);
for ins=1:total_ins
    for neighbor=2:total_ins-1 % First one, always the instance itself.
        diffs(ins, neighbor-1) = (sorted_corrs(ins, neighbor)-sorted_corrs(ins, neighbor+1))/sorted_corrs(ins, neighbor);
    end
end

% The neighborhood size k_i 
% is taken from the point of significant drop:
total_nbrs = 1;
for ins=1:total_ins
    num_neighbors = 1; % k_i
    for i=1:total_ins
        if diffs(ins,i)<threshold
            num_neighbors = num_neighbors+1;
            total_nbrs = total_nbrs+1;
        else
            break
        end
    end     
    nbrs_by_diff_corrs(ins,1) = num_neighbors;
end


gr_labels=cell(total_ins,1);
lambda_i=cell(total_ins,1);
lr_rate=0;
for i=1:total_ins
    % In some instances, the significant drop never happens. For these
    % instances, the closest instance is taken as the neighbor.
    if nbrs_by_diff_corrs(i)==0
        d = sorted_corrs(i,1);
        nn_labels = sorted_idxs(i,1);
    else
        d = sorted_corrs(i,2:nbrs_by_diff_corrs(i)+1);
        nn_labels = sorted_idxs(i,2:nbrs_by_diff_corrs(i)+1);               
    end
    % Control whether the neighbors' instances have positive silhouette
    % score or not.
    nn_labels_new=[];
    for ii=1:length(d)
        if ind_valsx(nn_labels(ii))
            nn_labels_new=[nn_labels_new,nn_labels(ii)];
        end
    end
    if isempty(nn_labels_new) && lambda ~=0
       if ind_valsx((i)) % If lambda==0, there is no need to check this
          nn_labels_new=sorted_idxs(i,1);
       end
    end
    gr_labels{i}=nn_labels_new;    
    if ~isempty(nn_labels_new)        
        if ind_valsx((i)) || lambda==1
            lambda_i{i}=(1-lambda)*(ones(1,length(nn_labels_new)))/length(nn_labels_new);
        else
           lambda_i{i}=(ones(1,length(nn_labels_new)))/length(nn_labels_new);
        end
    end
end

end