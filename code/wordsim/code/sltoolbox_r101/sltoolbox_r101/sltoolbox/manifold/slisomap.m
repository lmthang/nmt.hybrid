function [X, spectrum] = slisomap(G, d)
%SLISOMAP Performs ISOMAP manifold embedding
%
% $ Syntax $
%   - X = slisomap(G, d)
%   - [X, spectrum] = slisomap(G, d)
%
% $ Arguments $
%   - G:            The distance graph for neighbors
%   - d:            The dimension of embedded space
%   - X:            The embedded coordinates of the samples
%   - spectrum:     The eigenvalues of the embedded dimensions
%
% $ Description $
%   - X = slisomap(G, d) performs ISOMAP manifold embedding to pursue 
%     an embedding which best preserves the geodesic distances between
%     samples.
%
%   - [X, spectrum] = slisomap(G, d) additionally outputs the
%     eigen-spectrum of the embedded space.
%
% $ Remarks $
%   - In current implementation, the third-party toolbox: Matlab BGL is
%     required for computing the geodesics.
%
% $ History $
%   - Created by Dahua Lin, on Sep 8, 2006
%

%% parse and verify input

if nargin < 2
    raise_lackinput('slisomap', 2);
end

G = sladjmat(G, 'sparse', true);
n = size(G, 1);
if d >= n
    error('sltoolbox:exceedbound', ...
        'The embedded dimension d should be less than the number of samples');
end


%% compute

% compute geodesics
G = slsymgraph(G);
D = all_shortest_paths(G);

is_inf_dists = isinf(D);
is_inf_dists = is_inf_dists(:);
if any(is_inf_dists)
    error('sltoolbox:rterror', ...
        'The graph has multiple disconnected components');
end
clear is_inf_dists;

% perform MDS on D
[X, spectrum] = slcmds(D, d);

