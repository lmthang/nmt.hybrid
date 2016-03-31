function targets = sledgeset2adjlist(n, edges, sch)
%SLEDGESET2ADJLIST Converts edge set to adjacency list
%
% $ Syntax $
%   - targets = sledgeset2adjlist(n, edges, sch)
%
% $ Arguments $
%   - n:        The number of (source) nodes
%   - edges:    The set of edges (nedges x 2 or nedges x 3)
%   - sch:      The id of scheme of conversion to take
%               - 0:    no value -> no value
%               - 1:    no value -> has value
%               - 2:    has value -> no value
%               - 3:    has value -> has value
%   - targets:  The cell array of targets in adj list
%
% $ Remarks $
%   - an internal function for graph representation conversion.
%     no checking of input arguments would be performed.
%
% $ History $
%   - Created by Dahua Lin, on Sep 9, 2006
%

%% prepare storage

targets = cell(n, 1);

%% sort edges

I = edges(:, 1);
J = edges(:, 2);

[I, si] = sort(I);
J = J(si);

if sch == 3
    V = edges(si, 3);
elseif sch == 1
    nedges = length(I);
    V = ones(nedges, 1);
end

%% group targets

nums = slcount(I);
ni = length(nums);
[spos, epos] = slnums2bounds(nums);

if sch == 1 || sch == 3
    for i = 1 : ni
        s = spos(i); e = epos(i);
        if s <= e
            targets{I(s)} = [J(s:e), V(s:e)];
        end
    end
else
    for i = 1 : ni
        s = spos(i); e = epos(i);
        if s <= e
            targets{I(s)} = J(s:e);
        end
    end
end



