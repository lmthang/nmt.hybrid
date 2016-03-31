function A = sledges2adjmat(n, nt, edges, varargin)
%SLEDGES2ADJMAT Creates an adjacency matrix from edge set
%
% $ Syntax $
%   - A = sledges2adjmat(n, nt, edges, ...)
%
% $ Arguments $
%   - n:            The number of (source) nodes
%   - nt:           The number of (target) nodes
%   - edges:        The matrix of edge set
%   
% $ Description $
%   - A = sledges2adjmat(n, nt, edges) creates an adjacency matrix 
%     from the edge set. You can specify the following properties:
%       - 'valtype':        the value type of the target matrix
%                           - 'auto': if has value, make numeric matrix
%                                     if no value, make logical matrix
%                           - 'logical': make logical matrix always
%                           - 'numeric': make numeric matrix always
%                           (default = 'auto')
%       - 'sparse':         whether to create a sparse matrix 
%                           (default = true)
%       - 'preprune':       whether to prune the edges first 
%                           (default = false)
%       - 'prunemethod':    the method used to prune the edge set
%                           (default = [], means using default method)
%                           refer to slpruneedgeset for the specification 
%                           of the prune methods.
%       - 'sym':            whether to create symmetric graph
%       - 'symmethod':      the method to symmetrize the graph
%                           (default = [], means using default method)
%                           refer to slsymedgeset for the specification.
%
% $ Remarks $
%   - The property sym can only be true when n == nt.
%
%   - It is an integrated wrapper for slmakeadjmat, slsymgraph and
%     slpruneedgeset.
%
% $ History $
%   - Created by Dahua Lin, on Sep 9, 2006
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('sledges2adjmat', 3);
end

if ~isempty(edges)
    ncols = size(edges, 2);
    if ndims(edges) ~= 2 || (ncols ~= 2 && ncols ~= 3)
        error('sltoolbox:invalidarg', ...
            'The edges should be a 2D matrix with two or three columns');
    end
end

opts.valtype = 'auto';
opts.sparse = true;
opts.preprune = false;
opts.prunemethod = [];
opts.sym = false;
opts.symmethod = [];
opts = slparseprops(opts, varargin{:});

if opts.sym
    if n ~= nt
        error('sltoolbox:rterror', ...
            'The sym can only be true when n == nt');
    end
end

switch opts.valtype
    case 'auto'
        islogic = (ncols == 2);
    case 'numeric'
        islogic = false;
    case 'logical'
        islogic = true;
    otherwise
        error('sltoolbox:invalidarg', ...
        'Invalid value type of adjacency matrix: %s', opts.valtype);
end


%% main skeleton

% prune
if opts.preprune
    edges = slpruneedgeset(n, nt, edges, opts.prunemethod);
end

% make adjmat

A = slmakeadjmat(n, nt, edges, [], islogic, opts.sparse);

% symmetrize
if opts.sym
    A = slsymgraph(A, opts.symmethod);
end

