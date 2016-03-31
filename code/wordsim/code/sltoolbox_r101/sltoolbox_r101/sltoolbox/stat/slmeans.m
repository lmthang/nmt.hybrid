function varargout = slmeans(X, w, nums)
%SLMEANS Compute the mean vectors
%
% $ Syntax $
%   - v = slmeans(X)
%   - v = slmeans(X, w)
%   - V = slmeans(X, [], nums)
%   - V = slmeans(X, w, nums)
%   - [V, v] = slmeans(X, [], nums)
%   - [V, v] = slmeans(X, w, nums)
%
% $ Arguments $
%   - X:            the sample matrix with each column representing a sample
%   - w:            the sample weights
%   - nums:         the number of samples in each group
%   - v:            the overall mean vector
%   - V:            the matrix of mean vectors for each group
%
% $ Description $
%   - v = slmeans(X) computes the mean vector of X
%
%   - v = slmeans(X, w) computes the weighted mean vector of X, with weights
%     specified by w.
%
%   - V = slmeans(X, [], nums) computes the mean vectors for groups, 
%     each group of vectors are gathered together, and the number
%     of vectors in each collection is specified in nums
%
%   - V = slmeans(X, w, nums) computes the weighted mean vectors for 
%     groups of vectors.
%
%   - [V, v] = slmeans(X, w, nums) computes the mean vectors for groups
%     of vectors and the overall mean vector v.
%
%   - [V, v] = slmeans(X, w, nums) computes the weighted mean vectors for
%     groups of vectors and the overall weighted mean vector v.
%
% $ Remarks $
%   - If there are n samples in d-dimensional space, then X should be a 
%     d x n matrix, w should be a 1 x n row vector. If the vectors are
%     grouped in k groups, then nums should be a 1 x k row vector.
%     Then v would be a d x 1 vector, V would be a d x k matrix.
%
% $ History $
%   - Created by Dahua Lin on Dec 18th, 2005
%   - Modified by Dahua Lin on Apr 22nd, 2005
%       - extract the function computing mean of a group to an external
%         function.
%       - modify some comments
%

%% parse and verify input arguments
if ndims(X) ~= 2
    error('sltoolbox:invaliddims', 'X should be a 2D matrix');
end
[d, n] = size(X);

% for weights
if nargin < 2 || isempty(w)
    w = [];
else
    if ~isequal(size(w), [1, n])
        error('sltoolbox:sizmismatch', ...
            'the weight vector should be a 1 x n row vector');
    end
end

% for groupping
if nargin < 3 || isempty(nums)
    isgrouped = false;
else
    isgrouped = true;
    if size(nums, 1) ~= 1
        error('sltoolbox:invalidarg', ...
            'the nums vector should be a row vector');
    end
    if sum(nums) ~= n
        error('sltoolbox:sizmismatch', ...
            'the nums vector does not match the total number of vectors');
    end
    [sp, ep] = slnums2bounds(nums);  % group index boundary
    k = length(nums);  % number of groups
end

%% compute
if ~isgrouped
    v = slmean(X, w, true);
    varargout = {v};
else
    V = zeros(d, k);       
    
    % compute group-wise mean
    if isempty(w)
        for i = 1 : k
            V(:, i) = slmean(X(:, sp(i):ep(i)), [], true);
        end
    else
        for i = 1 : k
            V(:, i) = slmean(X(:, sp(i):ep(i)), w(sp(i):ep(i)), true);
        end
    end
    
    if nargout <= 1
        varargout = {V};
    else
        % compute group weights
        if isempty(w)
            gw = nums;
        else
            gw = zeros(1, k);
            for i = 1 : k
                gw(i) = sum(w(sp(i):ep(i)));
            end
        end
        
        % compute overall mean
        v = slmean(V, gw, true);
        
        varargout = {V, v};
    end
end



