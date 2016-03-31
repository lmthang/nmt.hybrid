function v = slmean(M, w, hasbeenchecked)
%SLMEAN Compute the mean vector of samples
%
% $ Syntax $
%   - v = slmean(M)
%   - v = slmean(M, w)
%   - v = slmean(M, w, true)
%
% $ Arguments $
%   - M:        the matrix of sample vectors stored as columns
%   - w:        the weights of the vectors
%   - v:        the computed mean vector
%
% $ Description $
%   - v = slmean(M) computes the mean vector of column vectors in M.
%   
%   - v = slmean(M, w) computes the weighted mean vector of column vectors
%     in M. w is the weights of the samples. if w is empty, the normal
%     mean vector would be computed. 
%
%   - v = slmean(M, w, true) indicates that the size consistency has been
%     checked by invoker. Then in this function, it will not be checked
%     again. This syntax is designed for the sake of efficiency.
%
% $ Remarks $
%   - M should be a 2D matrix (d x n), then w should be a 1 x n row vector,
%     v would be a d x 1 column vector.
%
% $ History $
%   - Created by Dahua Lin on Apr 22nd, 2006
%   - Modified by Dahua Lin on Sep 10th, 2006
%       - replace slmul by slmulvec to increase efficiency
%

%% parse and verify input arguments

if nargin < 3 || ~hasbeenchecked

    if ndims(M) ~= 2
        error('sltoolbox:invaliddims', 'M should be a 2D matrix');
    end

    n = size(M, 2); % number of samples
    if nargin < 2 || isempty(w)
        is_weighted = false;
    else
        is_weighted = true;

        % check size consistency
        [wd1, wd2] = size(w);
        if ndims(w) ~= 2 || wd1 ~= 1 || wd2 ~= n;
            error('sltoolbox:sizmismatch', ...
                'w is not a valid row vector consistent with M');
        end    
    end
    
else  
    n = size(M, 2); % number of samples
    is_weighted = ~isempty(w);
end


%% compute

if ~is_weighted    
    v = sum(M, 2) / n;    
else
    
    % normalize the weights
    w = w / sum(w);    
    v = sum(slmulvec(M, w, 2), 2);    
end

