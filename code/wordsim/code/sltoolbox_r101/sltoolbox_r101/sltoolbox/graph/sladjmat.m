function A = sladjmat(G, varargin)
%SLADJMAT Constructs the adjacency matrix representation of a graph
%
% $ Syntax $
%   - A = sladjmat(G, ...)
%
% $ Arguments $
%   - G:        The input graph
%   - A:        The adjacency matrix representation of the graph
%
% $ Description $
%   - A = sladjmat(G, ...) constructs the adjacency matrix representation
%     of a graph. You can specify the following properties to control
%     the construction.
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
% $ History $
%   - Created by Dahua Lin, on Sep 10, 2006
%

%% parse the graph

gi = slgraphinfo(G);



%% main skeleton

switch gi.form
    case 'edgeset'
        A = sledges2adjmat(gi.n, gi.nt, G.edges, varargin{:});
        
    case 'adjlist'
        if gi.valued
            sch = 3;
        else
            sch = 0;
        end
        edges = sladjlist2edgeset(G.targets, sch);
        A = sledges2adjmat(gi.n, gi.nt, edges, varargin{:});
        
    case 'adjmat'
        opts = struct(...
            'valtype', 'auto', ...
            'sparse', true, ...
            'preprune', false, ...
            'prunemethod', [], ...
            'sym', false, ...
            'symmethod', []);
        opts = slparseprops(opts, varargin{:});
        A = change_adjmat(G, opts);
end
   

%% auxiliary functions

function A = change_adjmat(A0, opts)

switch opts.valtype
    case 'auto'
        is_logic = islogical(A0);
    case 'logical';
        is_logic = true;
    case 'numeric'
        is_logic = false;
end

if is_logic
    if islogical(A0)
        A = A0;
    else
        A = (A0 ~= 0);
    end
else
    if isa(A0, 'double')
        A = A0;
    else
        A = double(A0);
    end
end

if opts.sparse
    if ~issparse(A)
        A = sparse(A);
    end
else
    if issparse(A)
        A = full(A);
    end
end

if opts.sym
    A = slsymgraph(A, opts.symmethod);
end


        
        
    

        







