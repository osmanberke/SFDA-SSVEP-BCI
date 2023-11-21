classdef ourClassificationLayer < nnet.layer.ClassificationLayer
    % Define custom classification layer to implement our proposed loss
    % function
    properties
        
    end
 
    methods

        function total_loss = forwardLoss(layer, Y, T)
            
            global lambda;
            global gr_labels;
            global lambda_i;
            global ind_valsx;
            
            sl_loss=crossentropy(Y(:,:,:,ind_valsx),T(:,:,:,ind_valsx),'DataFormat','UUUB'); % Calculate the Self-Adaptation Loss 
            
            
            ll_loss=0; % Initialize the Local-Regularity Loss

            % Calculate the Local-Regularity Loss
            for i=1:size(Y,4)                               
                for k=1:length(gr_labels{i,1})
                   ll_loss =ll_loss+ lambda_i{i,1}(k)*( crossentropy( Y(:,:,:,i), T(:,:,:,gr_labels{i,1}(k)),'DataFormat','UUUB') );
                end
            end
            total_loss=lambda*sl_loss+(ll_loss/size(Y,4)); % Calculate the total loss
        end
        
    end
end