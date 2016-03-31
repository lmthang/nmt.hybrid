function [X, spectrum] = slkernelembed(K, d, w)
%SLKERNELEMBED Finds an embedding space to preserve inner products
%
% $ Syntax $
%   - X = slkernelembed(K, d)
%   - X = slkernelembed(K, d, w)
%   - [X, info] = slkernelembed(...)
%
% $ Arguments $
%   - K:        The pairwise inner product matrix (kernel matrix): n x n
%   - d:        The dimension of the embedded space 
%   - w:        The weights of the samples: 1 x n
%   - X:        The embeded sample coordinates: d x n
%   - spectrum: The eigenvalues of the embeded space (dx1)
%
% $ Description $
%   - X = slkernelembed(K, d, w) finds an embedding space in which the
%     the inner product structure of the original sample set is best
%     preserved in terms of square error of reconstructed kernel matrix.
%
%   - X = slkernelembed(K, d, w) runs the algorithm on weighted samples.
%
%   - [X, spectrum] = slkernelembed(...) additionally outputs the
%     eigen-spectrum of the embedding space, which is a column vector
%     of eigenvalues associated with all dimensions.
%
% $ Remarks $
%   - The embeded dimension d should be less than the number of samples n.
%
%   - The implementation is based on EVD decomposition.
%
% $ History $
%   - Created by Dahua Lin, on Sep 8, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slkernelembed', 2);
end

if ndims(K) ~= 2 || size(K, 1) ~= size(K, 2)
    error('sltoolbox:invalidarg', ...
        'The K should be a square matrix');
end
n = size(K, 1);

if d >= n
    error('sltoolbox:exceedbound', ...
        'The dimension d should be less than the number of samples n');
end

if nargin < 3
    w = [];
else
    if ~isempty(w)
        if ~isequal(size(w), [1, n])
            error('sltoolbox:sizmismatch', ...
                'If w is specified, it should be an 1 x n row vector');
        end
    end
end


%% compute

% enforce symmetry
K = 0.5 * (K + K');

% assign weights
if ~isempty(w)
    for i = 1 : n
        K(i,:) = K(i,:) * w(i);
    end
    for i = 1 : n
        K(:,i) = K(:,i) * w(i);
    end
end

% decompose
[spectrum, X] = slsymeig(K, d);

spectrum = max(spectrum, 0);
s = sqrt(spectrum);
for i = 1 : d
    X(:,i) = X(:,i) * s(i);
end
X = X';

% de-weights
if ~isempty(w)
    for i = 1 : n
        X(:,i) = X(:,i) / w(i);
    end
end
    









   








