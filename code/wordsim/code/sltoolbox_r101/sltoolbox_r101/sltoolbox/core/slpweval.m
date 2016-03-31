function M = slpweval(X1, X2, f, varargin)
%SLPWEVAL Perform pairwise computation
%
% $ Syntax $
%   - M = slpweval(X1, X2, f)
%   - M = slpweval(X1, X2, f, ...)
%
% $ Arguments $
%   - X1:       the matrix of vectors serving as first argument of f
%   - X2:       the matrix of vectors serving as second argument of f
%   - f:        the function maps two vectors to a single scalar value
%   - M:        the matrix of pairwise evaluaton results
%
% $ Description $
%   - M = slpweval(X1, X2, f) takes vector arguments from X1 and X2, and 
%     computes the pairwise evaluation result with f. The vectors in X1
%     and X2 are stored in a column-wise manner. Suppose X1 and X2 have 
%     m and n columns respectively, then the resultant matrix M would be
%     of size m x n, with the M(i, j) = f(X1(:,i), X2(:,j)). 
%   
%   - M = slpweval(X1, X2, f, ...) conducts the computation with extra
%     parameters to f, i.e. M(i, j) = f(X1(:,i), X2(:,j), ...).
%
% $ Remarks $
%   - The vector length of the vectors in X1 and X2 are not necessarily
%     equal. The requirment on their dimensions depends on the callback
%     function f.
%   - For efficiency, the function would invoke f to evaluate in batch.
%     Thus f should support batch-evaluation. When the input arguments
%     to f have n columns, f should return an 1 x n row vector.
%
% $ History $
%   - Created by Dahua Lin on Apr 21, 2006
%   - Modified by Dahua Lin on Sep 10, 2006
%       - make some minor changes to suppress warnings
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('slpweval', 3);
end
[d1, n1] = size(X1);
[d2, n2] = size(X2);
slignorevars(d1, d2);

%% compute

% prepare output matrix
M = zeros(n1, n2);

if n1 > n2      % expand each column in X2 to n1 copies
 
    inds_e = ones(1, n1);
    for i = 1 : n2        
        x2 = X2(:, i);
        X2e = x2(:, inds_e);        
        M(:, i) = feval(f, X1, X2e, varargin{:})';            
    end
    
else            % expand each column in X1 to n2 copies
    
    inds_e = ones(1, n2);
    for i = 1 : n1
        x1 = X1(:, i);
        X1e = x1(:, inds_e);
        M(i, :) = feval(f, X1e, X2, varargin{:});
    end    
    
end

