function [Y, spectrum] = sllle_wg(G, d)
%SLLLE_WG Solves the Locally Linear Embedding from weight graph
%
% $ Syntax $
%   - Y = sllle_wg(G, d)
%   - [Y, spectrum] = sllle_wg(G, d)
%
% $ Arguments $
%   - G:        The weights graph
%   - d:        The dimension of the embedding
%   - Y:        The sample coordinates in the embedded space
%   - spectrum: The spectrum of the embeded space dimensions
%
% $ Description $
%   - Y = sllle_wg(G, d) solves the locally linear embedding from
%     a given weight graph. G should be a graph of n nodes, where
%     the edge value from i-th node to j-th node, means the weights
%     on the i-th sample in constructing the j-th sample, (or the
%     construction of i-th sample to the j-th sample). 
%     Y is solved by taking the eigenvectors of (I - W)(I - W)^T, 
%     corresponding to the (d+1) smallest eigenvalues, and discarding 
%     the smallest one.  In our output, the dimension is sorted in the 
%     ascending order of eigenvalues.
%
%   - [Y, spectrum] = sllle_wg(G, d) also returns the spectrum of the
%     corresponding dimensions, a column vector of the corresponding
%     eigenvalues.
%
% $ Remarks $
%   - The dimension d should be strictly less than n.
%
% $ History $
%   - Created by Dahua Lin, on Sep 11st, 2006
%

%% parse and verify input

if nargin < 2
    raise_lackinput('sllle_wg', 2);
end

gi = slgraphinfo(G, {'square'});
if ~gi.valued
    error('sltoolbox:invalidarg', 'The graph G should be a valued graph');
end
if isnumeric(G)
    W = G;
else
    W = sladjmat(G, 'sparse', true);
end

n = gi.n;
if d >= n
    error('sltoolbox:invalidarg', 'd should be strictly less than n');
end

%% compute

if issparse(W)
    M = speye(n) - W;
else
    M = eye(n) - W;
end
clear W;
M = M * M';

[spectrum, Y] = slsymeig(M, d+1, 'ascend');
clear M;
spectrum = spectrum(2:d+1);
Y = Y(:, 2:d+1);
Y = Y';


