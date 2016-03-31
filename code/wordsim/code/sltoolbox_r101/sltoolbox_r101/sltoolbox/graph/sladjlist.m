function Gd = sladjlist(G, uv)
%SLADJLIST Construct the adjacency list representation of a graph
%
% $ Syntax $
%   - Gd = sladjlist(G)
%   - Gd = sladjlist(G, uv)
%
% $ Arguments $
%   - G:        The input graph (or bigraph)
%   - Gd:       The output adjacency list representation of the graph
%   - uv:       The using-value scheme
%
% $ Description $
%   - Gd = sladjlist(G) constructs the adjacency list representation of the
%     graph G with an automatic selection of uv scheme.
%
%   - Gd = sladjlist(G, uv) constructs the adjacency list representation of 
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
        Gd.targets = sledgeset2adjlist(G.n, G.edges, sch);
        
    case 'adjlist'
        switch sch
            case {0, 3}
                Gd.targets = G.targets;
            case 1
                Gd.targets = cell(size(G.targets));
                nc = numel(G.targets);
                for i = 1 : nc
                    ci = G.targets{i};                  
                    if ~isempty(ci)
                        cn = size(ci, 1);
                        ci = [ci, ones(cn,1)];
                        Gd.targets{i} = ci;
                    end
                end
            case 2
                Gd.targets = cell(size(G.targets));
                nc = numel(G.targets);
                for i = 1 : nc
                    ci = G.targets{i};                  
                    if ~isempty(ci)
                        Gd.targets{i} = ci(:, 1);
                    end
                end
        end
        
    case 'adjmat'
        if gi.valued
            [I, J, V] = find(G);
            edges = [I, J, V];
            clear I J V;
        else
            [I, J] = find(G);
            edges = [I, J];
            clear I J;
        end
        Gd.targets = sledgeset2adjlist(size(G,1), edges, sch);
end

            
                








