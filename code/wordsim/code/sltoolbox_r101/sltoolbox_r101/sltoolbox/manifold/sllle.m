function [Y, spectrum, WG] = sllle(X, G, d, rwparams)
%SLLLE Performs Locally Linear Embedding
%
% $ Syntax $
%   - Y = sllle(X, G, d)
%   - Y = sllle(X, G, d, rwparams)
%   - [Y, spectrum, WG] = sllle(X, G, d)
%
% $ Arguments $
%   - X:        The sample matrix (d x n)
%   - G:        The neighborhood graph or the cell array of parameters to 
%               generate the neighborhood graph. 
%               For a graph, G should have n nodes, where n is the number
%               of samples in X. The non-zero of the entry (i,j) means 
%               the i-th sample is the neighbor of the j-th sample.
%               If it is a set of parameters, it is in the form of 
%               {method, ...}, which will be input to slfindnn to 
%               construct a neighborhood graph. 
%   - d:        The dimension of the embeded space. 
%   - rwparams: The cell array parameters for solving reconstruction weights
%               default = {}
%   - Y:        The coordinates of samples in the embeded space
%   - WG:       The weight graph computed
%
% $ Description $
%   - Y = sllle(X, G, d) solves the locally linear embedding of X in a 
%     d-dimensional linear space. 
%
%   - Y = sllle(X, G, d, rwparams) uses special set of parameters to 
%     control the solving of reconstruction parameters.
%
%   - [Y, spectrum, WG] = sllle(X, G, d) additionally returns the spectrum
%     and the local construction weight graph.
%
% $ Remarks $
%   - It integrates the functions: 
%       - slfindnn and slnngraph: for graph construction
%       - slreconweights: for local reconstruction weight solving
%       - sllle_wg: solves LLE from constructed weight graph
%   - Only the zero/non-zero of G takes effects, it will not make use
%     of the values in G.
%
% $ History $
%   - Created by Dahua Lin, on Sep 11st, 2006
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('sllle', 3);
end

if ~isnumeric(X) || ndims(X) ~= 2
    error('sltoolbox:invalidarg', ...
        'X should be a 2D numeric matrix');
end
n = size(X, 2);

if d >= n
    error('sltoolbox:invalidarg', ...
        'd should be strictly less than n');
end

if nargin < 4
    rwparams = {};
end


%% process the graph

if iscell(G)
    G = slnngraph(X, [], G, 'valtype', 'logical', 'sparse', true);
end

gi = slgraphinfo(G, {'square'});
if ~strcmp(gi.form, 'adjmat')
    G = sladjmat(G, 'valtype', 'logical', 'sparse', true);
end

%% solve reconstruction weights

WG = slnbreconweights(X, [], G, rwparams{:});
clear G;

%% solve the embedding

[Y, spectrum] = sllle_wg(WG, d);
    

    
    
    
    




