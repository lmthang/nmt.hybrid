function varargout = sltensor_svd(T, n)
%SLTENSOR_SVD Performs a Higher-Order SVD on a Tensor
%
% $ Syntax $
%   - [S, U1, U2, ...] = sltensor_svd(T)
%   - [S, U1, U2, ...] = sltensor_svd(T, n)
%   - [S, Us] = sltensor_svd(...)
%
% $ Description $
%   - [S, U1, U2, ...] = sltensor_svd(T) Performs a Higher Order Singular Value
%   Decomposition to a Tensor T. In the output arguments, S is the core
%   tensor, U1, U2, ... are the singular vector matrices of mode 1, 2,...
%
%   - [S, U1, U2, ...] = sltensor_svd(T, n) Here n specifies the order of the
%   tensor T, n should be not less than ndims(T). If n > ndims(T) is just
%   regarded as a tensor of dimensions d1xd2x...x1. 
%
%   - [S, Us] = sltensor_svd(...) where all mode matrices are returned to Us,
%   which is an 1 x n cell array, with each cell containing a mode matrix.
% 
% $ History $
%   - Created by Dahua Lin on Dec 17th, 2005
%

%% parse and verify the arguments
slchknargs(nargin, 1);
n0 = ndims(T);
if nargin == 1
    n = n0;
else
    if n < n0;
        error('sltoolbox:invalidarg', ...
            'The order %d is too small', n);
    end
end
if nargout > 2 && nargout ~= n+1
    error('sltoolbox:invalidnargout', ...
        'The number of outputs is not valid');
end

%% compute

Us = cell(n, 1);
S = T;
for i = 1 : n0
    M = sltensor_unfold(T, i);
    [d1, d2] = size(M);
    if d1 < d2
        [V, D, U] = svd(M', 0);
    else
        [U, D, V] = svd(M, 0);
    end
    slignorevars(D, V);
    clear D V;
    
    Us{i} = U;
    S = sltensor_multiply(S, U', i);
end
if n > n0
    for i = n0+1 : n
        Us{i} = 1;
    end
end

%% output
varargout{1} = S;
if nargout == 2
    varargout{2} = Us;
else
    varargout(2:n+1) = Us;
end


    






    
    