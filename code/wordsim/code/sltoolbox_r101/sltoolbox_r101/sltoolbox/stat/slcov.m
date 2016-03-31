function C = slcov(X, w, vmean, hasbeenchecked)
%SLCOV Compute the covariance matrix
%
% $ Syntax $
%   - C = slcov(X) 
%   - C = slcov(X, w)
%   - C = slcov(X, w, vmean)
%   - C = slcov(X, w, 0)
%   - C = slcov(X, w, vmean, true)
%
% $ Arguments $
%   - X:        the sample matrix
%   - w:        the weights of the samples
%   - vmean:    the pre-computed mean vector
%   - C:        the computed covariance matrix
%
% $ Description $
%   - C = slcov(X) computes the covariance matrix for the samples in X.
%     the samples are stored as column vectors. X should be a d x n 2D
%     matrix, where d is the vector dimension, and n is the number of
%     samples. 
%
%   - C = slcov(X, w) computes the weighted covariance matrix for
%     the samples in X. w is a 1 x n row vector of the sample weights.
%     If w is empty, the non-weighted covariance matrix is computed.
%
%   - C = slcov(X, w, vmean) computes the (weighted) covariance matrix
%     with the mean vector supplied. Thus in the function, vmean will be
%     used, instead of re-computing the mean vector. 
%
%   - C = slcov(X, w, 0) computes the (weighted) covariance matrix on 
%     the centered vectors. Since the vectors are treated as centered,
%     no mean vector would be computed, and X will not be shifted.
%
%   - C = slcov(X, w, vmean, true) indicates that the size consistency
%     has been verified by the invoker. In this function, no checking
%     will be conducted on the sizes of input arguments. This syntax
%     is designed for the sake of efficiency.
%
% $ Remarks $
%   - M should be a 2D matrix (d x n), then w should be a 1 x n row vector,
%     vmean should be a d x 1 column vector or 0. Then C would be a 
%     d x d matrix.
%   - In a non-weighted version, C = X * X' / n; while in a weighed
%     version, the weights would be first normalized before the covariance
%     is computed.
%
% $ History $
%   - Created by Dahua Lin on Apr 22, 2006
%   - Modified by Dahua Lin on Sep 10, 2006
%       - replace sladd by sladdvec to increase efficiency
%       - replace slmul by slmulvec
%

%% parse and verify input arguments

if nargin < 4 || ~hasbeenchecked

    % basic 
    if ndims(X) ~= 2
        error('sltoolbox:invaliddims', 'X should be a 2D matrix');
    end
    [d, n] = size(X);

    % for weights
    if nargin < 2
        w = [];
    end
    if ~isempty(w)
        if size(w, 1) ~= 1 || size(w, 2) ~= n
            error('sltoolbox:sizmismatch', ...
                'w should be an 1 x n row vector');
        end
    end

    % for mean vector
    if nargin >= 3 && ~isempty(vmean) && ~isequal(vmean, 0)
        if size(vmean, 1) ~= d || size(vmean, 2) ~= 1
            error('sltoolbox:sizmismatch', ...
                'v should be a d x 1 column vector');
        end
    end
    
else
    
    % simple parse
    n = size(X, 2);
    if nargin < 2
        w = [];
    end
    
end

%% centerize the samples

if nargin < 3 || isempty(vmean)
    vmean = slmean(X, w, true);
end

if ~isequal(vmean, 0) % need centerization
    X = sladdvec(X, -vmean, 1);
end

%% compute the covariance

if isempty(w)   % not weighted
    
    C = X * X' / n;
    
else            % weighted
    
    w = w / sum(w);    
    C = slmulvec(X, w, 2) * X';
        
end
    

    


