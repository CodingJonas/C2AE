function BW = compute_BW(X2)
% Calculates the bayesian network weights of the labels
    % load('mirflickr.mat');
    % X2 = [1 0 0; 1 1 1;0 1 0; 0 1 1];
    [N, L] = size(X2);
    BW = zeros(L,L);
    
    % Count coocurrences
    for i = 1:N
        % Check if each label is there, in that case, increment counts of
        % all other occurences
        current_instance = X2(i,:);
        for j = 1:L
            if current_instance(j)==1
                for k = j+1:L
                   if current_instance(k)==1
                       BW(j,k) = BW(j,k)+1;
                   end
                end
            end
        end
    end
    % sum(X2)' counts the total occurence of each label
    BW = (BW + BW')./sum(X2)';
    % heatmap(BW);
end
