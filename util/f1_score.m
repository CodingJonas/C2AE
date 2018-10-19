

function [precision, recall, F1] = f1_score(p, y)
	yval = y > 0;  % ground truth
	yhat = p > 0;  % predicted labels
    
    tp = sum(sum((yhat == 1) & (yval == 1)));
    fp = sum(sum((yhat == 1) & (yval == 0)));
    fn = sum(sum((yhat == 0) & (yval == 1)));

    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    F1 = (2 * precision * recall) / (precision + recall);
	

end

% Old code, calculating F1 wrong
% function [micro_f1, macro_f1] = f1_score(p, y)
% 	y = y > 0;
% 	p = p > 0;
% 	
% 	I = p .* y;  % TP
% 	U = p + y;  % TP + FP
% 	
%     % They only calculate precision
% 	micro_f1 = 2 * sum(sum(I)) / sum(sum(U));
% 	if isnan(micro_f1), micro_f1 = 1; end
% 	
% 	f = 2 * sum(I) ./ sum(U);
% 	f(isnan(f)) = [];
% 	macro_f1 = mean(f);
% end