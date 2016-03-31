function T2 = sltensor_multiply(T, varargin)
%SLTENSOR_MULTIPLY Multiplies a tensor and a matrix
%
% $ Syntax $
%   - T2 = sltensor_multiply(T, M, k)
%   - T2 = sltensor_multiply(T, Ms)
%   - T2 = sltensor_multiply(T, Ms, ks)
%   - T2 = sltensor_multiply(T, M1, k1, M2, k2, ...)
%
% $ Description $
%   - T2 = sltensor_multiply(T, M, k) Computes the tensor multiplication between
%   a tensor and a 2D matrix along the k-th mode.
%
%   - T2 = sltensor_multiply(T, Ms) Sequentially multiplies the tensor with
%   the elements in Ms, which is a cell array from the mode 1 to n. n is
%   the number of matrices in Ms. It should be that n == ndims(T).
%
%   - T2 = sltensor_multiply(T, Ms, ks) Sequentially multiplies the tensor
%   with the matrices in the cell array Ms along the modes specified in
%   corresponding element in ks. ks should be an array with the same size
%   as Ms.
%
%   - T2 = sltensor_multiply(T, M1, k1, M2, k2, ...) Sequentially multiplies 
%   the tensor with the matrices M1, M2, ... along the modes k1, k2, ...
%
% $ History $
%   - Created by Dahua Lin on Dec 17th, 2005
%

%% parse and verify input arguments
if nargin < 2
    raise_lackinput('sltensot_multiply', 2);
end
n0 = ndims(T);
if iscell(varargin{1});
    Ms = varargin{1};
    if nargin == 2
        n = numel(Ms);
        if n ~= n0
            error('sltoolbox:invalidarg', ...
                'For the case no specifying mode indices, it should be n == ndims(T)');
        end
        ks = 1:n;
    elseif nargin == 3
        ks = varargin{2};
        if ~isequal(size(Ms), size(k))
            error('sltoolbox:argmismatch', ...
                'The size of ks does not match that of Ms');
        end    
    else
        error('sltoolbox:invalidarg', ...
            'Invalid input arguments');
    end
else
    if mod(length(varargin), 2) ~= 0
        error('sltoolbox:invalidarg', ...
            'Invalid input arguments');
    end
    Ms = varargin(1:2:end);
    ks = [varargin{2:2:end}];
    if numel(Ms) ~= numel(ks)
        error('sltoolbox:invalidarg', ...
            'Invalid input arguments');
    end
end
if any(ks(:) < 1)
    error('sltoolbox:invalidarg', ...
        'mode indices should all be positive integers');
end
maxk = max(ks(:));
dims = size(T);
if maxk > n0
    dims = [dims, ones(1, maxk-n0)];
end
nmat = numel(ks);
    
%% compute
if nmat == 1
    T2 = multiply_tensor_matrix(T, Ms{1}, ks, dims);
else
    T2 = T;
    for i = 1 : nmat
        T2 = multiply_tensor_matrix(T2, Ms{i}, ks(i), dims);
    end
end
    


%% compute (by converting to matrix product)

function T2 = multiply_tensor_matrix(T, M, k, dims)

Tk = sltensor_unfold(T, k);
T2 = M * Tk;
clear Tk;
dims(k) = size(M, 1);
T2 = sltensor_fold(T2, dims, k);





