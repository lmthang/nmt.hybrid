function gi = slgraphinfo(G, conds)
%SLGRAPHINFO Extracts basic information of a given graph representation
%
% $ Syntax $
%   - gi = slgraphinfo(G)
%   - gi = slgraphinfo(G, conds)
%
% $ Arguments $
%   - G:        The input graph
%   - conds:    The cell array of conditions to be checked
%   - gi:       the information of the graph
%               a struct with the following fields:
%               - type: the type of the graph:
%                       - 'ge': a general graph
%                       - 'bi': a bigraph
%               - form: the form of the graph representation:
%                       - 'edgeset': edget set representation
%                       - 'adjlist': adjacency list representation
%                       - 'adjmat':  adjacency matrix representation
%               - n:    the number of (source) nodes
%               - nt:   the number of (target) nodes
%               - valued: whether the graph has values on edges
%
% $ Description $
%   - gi = slgraphinfo(G) extracts the basic information of the graph
%     representation and examines the integrity of it. 
%
%   - gi = slgraphinfo(G, conds) verifies whether the graph conforms
%     to special conditions. The acceptable conditions:
%       - 'square': a square graph, it can be a general graph or a bigraph
%                   with n == nt. All graph that is square can be
%                   considered as a general graph
%       - 'edgeset': the representation is an edge set
%       - 'adjlist': the representation is an adjacency list
%       - 'adjmat':  the representation is an adjacency matrix
%       - [n]:       the n is equal to specified value
%       - [n, nt]:   the n and nt are equal to specified value
%       - 'numeric': the value type is numeric (only take effect for adjmat)
%       - 'logical': the value type is logical (only take effect for adjmat)
%
% $ Remarks $
%   - For a general graph, n and nt are equal, being the number of nodes.
%
%   - When multiple conditions on representation form are specified, only
%     the last one takes effect.
%
% $ History $
%   - Created by Dahua Lin, on Sep 9, 2006
%

%% parse conditions

if nargin < 2 || isempty(conds)
    cc = {};
    fc = [];
else
    nconds = length(conds);
    tf = false(1, nconds);
    pf = 0;
    for i = 1 : nconds
        curcond = conds{i};
        if ischar(curcond)
            switch curcond 
                case {'adjmat', 'edgeset', 'adjlist'}
                    tf(i) = 1;
                    pf = i;
            end
        end
    end
    if pf > 0
        fc = conds{pf};
    else
        fc = [];
    end
    cc = conds(~tf);    
end


%% parse representation form and delegate

if isnumeric(G) || islogical(G)
    
    if ~isempty(fc)
        chk_form('adjmat', fc);
    end
    gi = ginfo_adjmat(G);
    
elseif isstruct(G)
    if isfield(G, 'edges') && ~isfield(G, 'targets')
        
        if ~isempty(fc)
            chk_form('edgeset', fc);
        end
        gi = ginfo_edgeset(G);
        
    elseif isfield(G, 'targets') && ~isfield(G, 'edges')
        
        if ~isempty(fc)
            chk_form('adjlist', fc);
        end
        gi = ginfo_adjlist(G);
        
    else
        report_unreg();
    end
else
    report_unreg();
end


%% check conditions

if ~isempty(cc)
    ncc = numel(cc);
    for i = 1 : ncc
        chk_cond(G, gi, cc{i});
    end
end



%% Information function for different forms

function gi = ginfo_edgeset(G)

gi.form = 'edgeset';

if ~isfield(G, 'n') || ~isfield(G, 'edges')
    chkerr('The edgeset representation of a graph should have the fields n and edges');
end

ncols = size(G.edges, 2);
if ~isempty(G.edges) && ncols ~= 2 && ncols ~= 3
    chkerr('The non-empty edges should have 2 or 3 columns');
end

if isfield(G, 'nt')
    gi.type = 'bi';
    gi.n = G.n;
    gi.nt = G.nt;        
    
else
    gi.type = 'ge';
    gi.n = G.n;
    gi.nt = G.n;
    
end

gi.valued = (ncols == 3);


function gi = ginfo_adjlist(G)

gi.form = 'adjlist';

if ~isfield(G, 'n') || ~isfield(G, 'targets')
    chkerr('The adjlist representation of a graph should have the fields n and targets');
end

tars = G.targets;
if ~iscell(tars) 
    chkerr('The targets should be a cell array');
end
if ~isequal(size(tars), [1, G.n]) && ~isequal(size(tars), [G.n, 1])
    chkerr('The targets should be a cell array of size 1 x n or n x 1');
end        

if isfield(G, 'nt')
    gi.type = 'bi';
    gi.n = G.n;
    gi.nt = G.nt;        
    
else
    gi.type = 'ge';
    gi.n = G.n;
    gi.nt = G.n;
    
end

n = G.n;
gi.valued = false;
for i = 1 : n
    if ~isempty(tars{i})
        ncols = size(tars{i}, 2);
        gi.valued = (ncols > 1);
        break;
    end
end
    


function gi = ginfo_adjmat(G)

gi.form = 'adjmat';

gi.n = size(G, 1);
gi.nt = size(G, 2);

if gi.n == gi.nt
    gi.type = 'ge';
else
    gi.type = 'bi';
end

gi.valued = isnumeric(G);



%% Condition checking function

function chk_cond(G, gi, cond)

if ischar(cond)
    switch cond
        case 'square'
            if gi.n ~= gi.nt
                conderr('The graph is square');
            end
        case 'logical'
            if isnumeric(G)
                conderr('The value type is logical');
            end
        case 'numeric'
            if islogical(G)
                conderr('The value type is numeric');
            end
        otherwise
            error('sltoolbox:invalidarg', ...
                'Unknown condition: %s', cond);
    end
else
    if isscalar(cond)
        if gi.n ~= cond
            conderr(sprintf('n = %d', cond));
        end
    elseif isvector(cond) && length(cond) == 2
        if gi.n ~= cond(1) || gi.nt ~= cond(2)
            conderr(sprintf('n = %d, nt = %d', ...
                cond(1), cond(2)));
        end
    else
        error('sltoolbox:invalidarg', ...
            'Invalid condition is specified');
    end
end
    

%% Auxiliary function

function chk_form(df, cf)

if ~strcmp(df, cf)
    error('sltoolbox:invalidarg', ...
        'The given graph is like the form %s, which is not the required form %s', ...
        df, cf);
end


function report_unreg()

error('sltoolbox:invalidarg', ...
    'The graph form is unrecognizable');

function chkerr(msg)

error('sltoolbox:invalidarg', ...
    'Invalid graph representation: %s', msg);

function conderr(msg)

error('sltoolbox:invalidarg', ...
    'The given graph does not meet the requirement: %s', msg);






    



