function varargout = slcovs(X, w, nums, M)
%SLCOVS Computes the sample covariance matrix
%
% $ Syntax $
%   - C = slcovs(X)
%   - C = slcovs(X, w)
%   - Cs = slcovs(X, [], nums)
%   - Cs = slcovs(X, w, nums)
%   - C = slcovs(X, w, [], M)
%   - Cs = slcovs(X, w, nums, M)
%   - [Cs, Cpool] = slcovs(...)
%
% $ Arguments $
%   - X:           the sample matrix with each column representing a sample
%   - w:           the sample weights
%   - nums:        the number of samples in the groups
%   - M:           the means or related information
%   - C:           the overall covariance matrix
%   - Cs:          the covariance matrices for the groups
%   - Cpool:       the pooled covariance matrix
%
% $ Description $
%   - C = slcovs(X) computes the covariance matrix for the samples in X.
%
%   - C = slcovs(X, w) computes the weighted covariance matrix for the 
%     samples in X, with weights specified in w.
%
%   - C = slcovs(X, [], nums) computes the covariance matrices for 
%     collections of samples, with the number of samples in each 
%     collections specified by the row vector nums.
%
%   - C = slcovs(X, w, nums) computes the weighted covariance matrices for
%     collections of samples, with the sample weights specified in w,
%     and the number of samples in collections specified in nums.
%
%   - C = slcovs(X, w, [], M) also provides mean information via M. M = 0
%     indicates that the mean vector is certainly zero vector. M can
%     also be a d x 1 row vector specifying the (weighted) mean vector
%     of X. If w is [], then all samples are treated equally.
%         
%   - Cs = slcovs(X, w, nums, M) computes the (weighted) covariance matrices 
%     for collections of vectors. When computing normal covariance without 
%     weighting you can set w = []. M = 0 indicates that the (weighted) mean 
%     vector for all groups of samples are zero vectors. M can be a 
%     d x k matrix storing the (weighted) mean vectors for all groups 
%     of samples.
%
%   - [Cs, Cpool] = slcovs(...) computes the pool covariance matrices for
%     all collections of vectors. If the samples is not groupped, then
%     Cpool is equal to Cs.
%
% $ History $
%   - Created by Dahua Lin on Dec 18th, 2005
%

%% parse and verify input arguments

% basic
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
    k = 1;
else
    isgrouped = true;
    if size(nums, 1) ~= 1 
        error('sltoolbox:sizmismatch', ...
            'the nums vector should be a row vector');
    end
    if sum(nums) ~= n
        error('sltoolbox:sizmismatch', ...
            'the nums vector does not match the total number of vectors');
    end    
    k = length(nums);
end

% for mean vectors
if nargin < 4
    M = [];
elseif ~isempty(M) && ~isequal(M, 0) 
    if ~isequal(size(M), [d, k])        
        error('sltoolbox:sizmismatch', ...
            'the matrix of means should be a d x k matrix');
    end
end


%% compute the covariances

if ~isgrouped      % not groupped
    
    % compute the covariance
    C = slcov(X, w, M);
    
    % output
    if nargout == 1
        varargout = {C};
    elseif nargout == 2
        varargout = {C, C};
    end
           
else                % groupped
    
    % compute groupwise covariances
    Cs = compute_groupwise_covariances(nums, d, k, X, w, M);
    
    % output
    if nargout == 1        
        varargout = {Cs};        
    elseif nargout == 2        
        % compute pooled covariance
        Cpool = compute_pooled_covariance(Cs, d, k, nums, w);                        
        varargout = {Cs, Cpool};
    end
    
end
 

%% Sub functions


%% the function for computing group-wise covariance matrices
function Cs = compute_groupwise_covariances(nums, d, k, X, w, M)

[sp, ep] = slnums2bounds(nums);

% prepare storage
Cs = zeros(d, d, k);

if isempty(M)           % no precomputed means

    if isempty(w)
        for i = 1 : k
            Cs(:, :, i) = slcov(X(:, sp(i):ep(i)), [], []);
        end
    else
        for i = 1 : k
            Cs(:, :, i) = slcov(X(:, sp(i):ep(i)), w(sp(i):ep(i)), []);
        end
    end
   
elseif isequal(M, 0)        % pre centralized

    if isempty(w)
        for i = 1 : k
            Cs(:, :, i) = slcov(X(:, sp(i):ep(i)), [], 0);
        end
    else
        for i = 1 : k
            Cs(:, :, i) = slcov(X(:, sp(i):ep(i)), w(sp(i):ep(i)), 0);
        end
    end
      
else                        % precomputed means

    if isempty(w)
        for i = 1 : k
            Cs(:, :, i) = slcov(X(:, sp(i):ep(i)), [], M(:, i));
        end
    else
        for i = 1 : k
            Cs(:, :, i) = slcov(X(:, sp(i):ep(i)), w(sp(i):ep(i)), M(:, i));
        end
    end

end


%% the function for computing pooled covariance
function Cpool = compute_pooled_covariance(Cs, d, k, nums, w)

[sp, ep] = slnums2bounds(nums);

% compute group weights
if isempty(w)
    gw = nums;
else
    gw = zeros(1, k);
    for i = 1 : k
        gw(i) = sum(w(sp(i):ep(i)));
    end
end
gw = gw(:) / sum(gw);

% compute with reshape
Cs = reshape(Cs, [d*d, k]);
Cpool = Cs * gw;
Cpool = reshape(Cpool, [d, d]);



