% loss function used in our paper
% p is the output of the network
% y is the true labels
% w is, what the heck is w?
function [E, G] = BR_error(p, y, w, BW)
    if gpuDeviceCount>1
        p = gather(p);
        y = gather(y);
    end
    
    % Check if y contains any negative label. If not, make some negativ by
    % substracting 0.5
    if sum(sum(y < 0)) == 0
        y = y - 0.5;
    end
    [N, D] = size(y);
    % Divide y and p into single cells for each training instance
    pc = mat2cell(p, ones(1, N), D);
    yc = mat2cell(y, ones(1, N), D);
    [Ec, Gc] = cellfun(@(a, b) error_new(a, b, w, BW), pc, yc, 'UniformOutput', false);
    E = mean(cell2mat(Ec));
    G = cell2mat(Gc);
    if gpuDeviceCount>1
        E = gpuArray(E);
        G = gpuArray(G);
    end
end

function [e, g] = error(p, y, w, BW)
    YEE = (y > 0);  % right labels
    LIN = (y < 0);  % wrong labels
    weight = w(LIN, YEE);
    num = sum(YEE) * sum(LIN);
    p1 = p(YEE);    % probabilities right labels
    p0 = p(LIN);    % probabilites wrong labels
    err = bsxfun(@minus, p1, p0.');
    % 5 is a fixed parameter in our loss function, we fix the parameter
    %througout all experiments
    err =  exp(-5 * err) ./ num;
    e = sum(sum(err));
    
    % Set the gradient
    g = zeros(size(y));
    g(YEE) = -sum(err, 1);
    g(LIN) = sum(err, 2).';
end

function [e, g] = error_new(p, y, w, BW)
    power = 5;  % Used as a factor in the exponential function
    YEE = (y > 0);  % right labels
    LIN = (y < 0);  % wrong labels
    weight = w(LIN, YEE);
    num = sum(YEE) * sum(LIN);
    p1 = p(YEE);    % probabilities right labels
    p0 = p(LIN);    % probabilites wrong labels
    
    % Calculate ranking error
    % 5 is a fixed parameter in our loss function, we fix the parameter
    %througout all experiments
    e_r = bsxfun(@minus, p1, p0.');
    e_r =  exp(-1 * power * e_r) ./ num;
    
    % Set the gradient
    g = zeros(size(y));
    g(YEE) = -sum(e_r, 1);
    g(LIN) = sum(e_r, 2).';
    
    
    % Calculate causation error for TP
    lambda = 10.0;
    indexes_true = find(y>0);
    num_true = length(indexes_true);
    e_ctp = zeros(length(y),1);
    for i = indexes_true
        % sum up error for output i
        for j = indexes_true
            e_ctp(i) = e_ctp(i) + BW(i,j) * exp(power * (p(i) - p(j))) - BW(j,i) * exp(power * (p(j) - p(i)));
            %e_c(i) = e_c(i) + BW(i,j)*(1-p(j)) - BW(j,i)*p(j);
        end
        e_ctp(i) = e_ctp(i)/(num_true^2);
        g(i) = g(i) + lambda * e_ctp(i);
    end
    
    % Calculate causation error for TN
    lambda = 1.0;
    indexes_false = find((p>0.5 & LIN)==1);
    num_false = length(indexes_false);
    e_ctn = zeros(length(y),1);
    
    for i = indexes_true
        for j = indexes_false
            e_ctn(i) = e_ctn(i) + (1-BW(i,j)) * exp(power * (p(i)-(1-p(j))));
        end
        e_ctn(i) = e_ctn(i)/(num_true*num_false+1);
        g(i) = g(i) + lambda * e_ctn(i);
    end
    
    for j = indexes_false
        for i = indexes_true
            e_ctn(j) = e_ctn(j) + (1-BW(j,i)) * exp(power * (p(j)-(1-p(i))));
        end
        e_ctn(j) = e_ctn(j)/(num_true*num_false+1);
        g(j) = g(j) + lambda * e_ctn(j);
    end
    
    
    % Calculate overall error
    e = sum(sum(e_r)) + sum(sum(e_ctp)) + sum(sum(e_ctn));
    e = sum(sum(e_r)) + sum(sum(e_ctp));
end
