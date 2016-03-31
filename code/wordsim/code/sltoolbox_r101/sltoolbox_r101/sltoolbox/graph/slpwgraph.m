function G = slpwgraph(Xs, Xt, n, nt, evalfunctor, varargin)
%SLVALGRAPH Constructs a graph by computing values between nodes pairwisely
%
% $ Syntax $
%   - G = slpwgraph(X, Xt, n, nt, evalfunctor, ...)
%
% $ Arguments $
%   - X:            The set of (source) nodes
%   - Xt:           The set of (target) nodes
%   - n:            The number of (source) nodes
%   - nt:           The number of (target) nodes
%   - evalfunctor:  The functor to evaluate the values between two sets of
%                   nodes, it should be like the following form:
%                   V = f(X, Xt, inds1, inds2, ...)
%                   Here, inds1 and inds2 are the indices selected in the
%                   subset for current batch of computation. If inds1 and
%                   inds2 respectively refer to n1 and n2 samples, then
%                   V should be a n1 x n2 matrix (full or sparse)
%   - G:            The constructed graph
%
% $ Description $
%   - G = slpwgraph(X, Xt, n, nt, evalfunctor, ...) constructs a adjacency 
%     matrix of a graph by computing edge values between every pair of the 
%     nodes. If Xt is empty, then Xt is considered as the same as X,
%     nt is considered as equal to n.
%     You can specify the following properties:
%     \*
%     \t   The Properties of Graph Matrix construction           \\
%     \h      name     &      description
%            'sparse'  & whether the target graph G is sparse 
%                        (default = true)
%            'valtype' & The type of values in G: 'logical'|'numeric'
%                        (default = 'numeric')
%                        The value output by evalfunctor should conform
%                        to the specified valtype
%            'maxblk'  & The maximum number of elements that can be
%                        computed in each batch. (default = 1e7) 
%     \*
%  
% $ History $
%   - Created by Dahua Lin, on Sep 8th, 2006
%   - Modified by Dahua Lin, on Sep 10th, 2006
%       - Use new graph construction functions
%       - Add support for bigraph
%

%% parse and verify input arguments

if nargin < 5
    raise_lackinput('slpwgraph', 5);
end

if isempty(Xt)
    Xt = Xs;
    nt = n;
end

tarsiz = [n, nt];

opts.sparse = true;
opts.valtype = 'numeric';
opts.maxblk = 1e7;
opts = slparseprops(opts, varargin{:});

if ~ismember(opts.valtype, {'logical', 'numeric'})
    error('sltoolbox:invalidarg', ...
        'Invalid value type for graph: %s', opts.valtype);
end


%% compute

% create partitions

ps = slequalpar2D(tarsiz, opts.maxblk);
nm = length(ps(1).sinds);
nn = length(ps(2).sinds);

% compute

if opts.sparse
    nblks = nm * nn;
    CI = cell(nblks, 1);
    CJ = cell(nblks, 1);
    CV = cell(nblks, 1);
    
    k = 0;    
    for i = 1 : nm
    for j = 1 : nn
        
        % get indices
        k = k + 1;
        inds1 = ps(1).sinds(i):ps(1).einds(i);
        inds2 = ps(2).sinds(j):ps(2).einds(j);
        
        % compute
        curV = slevalfunctor(evalfunctor, Xs, Xt, inds1, inds2);
        
        % filter
        [curI, curJ, curV] = find(curV);
        curI = curI + (inds1(1) - 1);
        curJ = curJ + (inds2(1) - 1);
        
        % store
        CI{k} = curI;
        CJ{k} = curJ;
        CV{k} = curV;
        
    end
    end
    
    CI = vertcat(CI{:});
    CJ = vertcat(CJ{:});
    CV = vertcat(CV{:});
    
    edges = [CI, CJ];
    clear CI CJ;
    
    islogic = strcmp(opts.valtype, 'logic');
    
    G = slmakeadjmat(n, nt, edges, CV, islogic, true);
                        
else
    
    switch opts.valtype
        case 'logical'
            G = false(n, nt);
        case 'numeric'
            G = zeros(n, nt);
    end
    
    for i = 1 : nm
    for j = 1 : nn
        
        % get indices
        inds1 = ps(1).sinds(i):ps(1).einds(i);
        inds2 = ps(2).sinds(j):ps(2).einds(j);
        
        % compute
        curV = slevalfunctor(evalfunctor, Xs, Xt, inds1, inds2);
        
        % store
        G(inds1, inds2) = curV;                
        
    end
    end
        
end

                







