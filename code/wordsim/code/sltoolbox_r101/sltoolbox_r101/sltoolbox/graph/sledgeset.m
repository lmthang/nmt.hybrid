function Gd = sledgeset(G, uv)
%SLEDGESET Construct the edge set representation of a graph
%
% $ Syntax $
%   - Gd = sledgeset(G)
%   - Gd = sledgeset(G, uv)
%
% $ Arguments $
%   - G:        The input graph (or bigraph)
%   - Gd:       The output edge set representation of the graph
%   - uv:       The using-value scheme
%
% $ Description $
%   - Gd = sledgeset(G) constructs the edge set representation of the
%     graph G with an automatic selection of uv scheme.
%
%   - Gd = sledgeset(G, uv) constructs the edge set representation of 
%     the graph G with a specified uv scheme.
%     There are the following uv schems:
%       - 'auto':   automatic selection (default)
%                   If G has values then use the values for construction,
%                   otherwise not use.
%       - 'on':     force to use values, if G has no values then assign
%                   1 to all edges
%       - 'off':    force not to use values even if G has values.
%
% $ History $
%   - Created by Dahua Lin, on Sep 9, 2006
%

%% parse and verify input

gi = slgraphinfo(G);

if nargin < 2 || isempty(uv)
    uv = 'auto';
end

%% decide scheme
% sch:
%   0:  no value -> no value
%   1:  no value -> has value
%   2:  has value -> no value
%   3:  has value -> has value

switch uv
    case 'auto'
        if gi.valued
            sch = 3;
        else
            sch = 0;
        end
    case 'on'
        if gi.valued
            sch = 3;
        else
            sch = 1;
        end
    case 'off'
        if gi.valued
            sch = 2;
        else
            sch = 0;
        end
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid using-value scheme: %s', uv);
end

%% do construction

switch gi.type
    case 'ge'
        Gd = struct('n', gi.n);
    case 'bi'
        Gd = struct('n', gi.n, 'nt', gi.nt);
end

switch gi.form
    case 'edgeset'
        switch sch
            case {0, 3}
                Gd.edges = G.edges;
                
            case 1
                if ~isempty(G.edges)
                    vals = ones(size(G.edges, 1), 1);
                    Gd.edges = [G.edges, vals];
                else
                    Gd.edges = [];
                end
            case 2
                if ~isempty(G.edges)
                    Gd.edges = G.edges(:, 1:2);
                else
                    Gd.edges = [];
                end
        end
        
    case 'adjlist'
        Gd.edges = sladjlist2edgeset(G.targets, sch);
        
    case 'adjmat'
        switch sch
            case {0, 2}
                [I, J] = find(G);
                Gd.edges = [I, J];
            case 1
                [I, J] = find(G);
                ne = length(I);
                Gd.edges = [I, J, ones(ne, 1)];
            case 3
                [I, J, V] = find(G);
                Gd.edges = [I, J, V];
        end
end
                
  
