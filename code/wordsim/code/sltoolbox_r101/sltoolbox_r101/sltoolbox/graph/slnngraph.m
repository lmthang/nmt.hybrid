function G = slnngraph(X, X2, nnparams, varargin)
%SLNNGRAPH Constructs a nearest neighborhood based graph
%
% $ Syntax $
%   - G = slnngraph(X, [], nnparams, ...)
%   - G = slnngraph(X, X2, nnparams, ...)
% 
% $ Arguments $
%   - X:        The sample matrix of the (referenced) nodes
%   - X2:       The sample matrix of the (query) nodes
%   - nnparams: The cell array of parameters to slfindnn for determining
%               neighborhoods.
%   - G:        The adjacency matrix of the constructed graph
%
% $ Description $
%   - G = slnngraph(X, [], nnparams, ...) constructs a graph with adjacency
%     matrix representation using specified nearest neighbor strategy.
%     You can further specify the following properties to control the
%     process of adjacency matrix construction.
%       \*
%       \t   The Properties of Graph Matrix construction           \\
%       \h     name         &      description                      \\
%             'valtype'     & The type of the matrix values          \\
%                             - 'logical':  using logical value
%                             - 'numeric':  using numeric value (default)
%             'sparse'      & Whether to construct sparse matrix 
%                             (default = true)                      \\
%             'tfunctor'    & The function to transform the distance
%                             values to edge values. (default = [])  \\
%             'sym'         & whether to symmetrizes the graph 
%                             (default = false)                       \\
%             'symmethod'   & The method used to symmetrization       
%                             (default = [])                          \\
%       \*
%
%   - G = slnngraph(X, X2, nnparams, ...) constructs a bigraph with the
%     source and target set respectively specified.
%
%   - There are three configurations on X and X2:
%       - construct graph with the same set X with all edges connecting
%         the same node excluded. You can use the following syntax:
%           slnngraph(X, [], ...)
%       - construct graph with the same set X with the edges connecting
%         the same node preserved. You can use the following syntax:
%           slnngraph(X, X, ...)
%       - construct graph with the different source and target sets X and
%         X2, You can use the following syntax:
%           slnngraph(X, X2, ...)
%
% $ Remarks $
%   - tfunctor only takes effect for numeric value type.
%
%   - The option sym and symmethod only take effect when X and X2 has the
%     same number of elements.
%
% $ History $
%   - Created by Dahua Lin, on Sep 8th, 2006
%   - Modified by Dahua Lin, on Sep 11st, 2006
%       - revise to conform to the defined neighborhood system
%         representation
%       - fix small bugs
%

%% parse input

if nargin < 3
    raise_lackinput('slnngraph', 3);
end

if ~isnumeric(X) || ndims(X) ~= 2 
    error('sltoolbox:invalidarg', ...
        'X should be a 2D numeric matrix');
end

if isempty(X2)
    X2 = [];
else
    if ~isnumeric(X2) || ndims(X2) ~= 2 
        error('sltoolbox:invalidarg', ...
            'A non-empty X2 should be a 2D numeric matrix');
    end
end
    
if ~iscell(nnparams)
    error('stoolbox:invalidarg', ...
        'The parameters for slfindnn should be groupped in a cell array');
end

opts.valtype = 'numeric';
opts.sparse = true;
opts.tfunctor = [];
opts.sym = false;
opts.symmethod = [];
opts = slparseprops(opts, varargin{:});

switch opts.valtype
    case 'logical'
        islogic = true;
    case 'numeric';
        islogic = false;
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid value type for graph: %s', opts.valtype);
end


%% Main 

n0 = size(X, 2);
if isempty(X2)
    nq = n0;    
else
    nq = size(X2, 2);
end
can_sym = (n0 == nq);

% find nearest neighbor
if ~islogic
    [nnidx, dists] = slfindnn(X, X2, nnparams{:});
else
    nnidx = slfindnn(X, X2, nnparams{:});
end

% group edges and vectorizes distances
edges = sladjlist2edgeset(nnidx, 0);
edges = edges(:, [2,1]);  % flip in order to place source to left, target to right
clear nnidx;
if ~islogic
    dists = vertcat(dists{:});
else
    dists = [];
end

% compute edge values
if ~isempty(dists)
    if isempty(opts.tfunctor)
        vals = dists;
    else
        vals = slevalfunctor(opts.tfunctor, dists);
    end
else
    vals = [];
end
clear dists;

% make graph matrix
G = slmakeadjmat(n0, nq, edges, vals, islogic, opts.sparse);

% symmetrize the graph 
if opts.sym && can_sym
    G = slsymgraph(G, opts.symmethod);
end



  