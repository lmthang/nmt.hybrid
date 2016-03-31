function edges = sladjlist2edgeset(targets, sch)
%SLADJLIST2EDGESET Converts the adjacency list to edge set
%
% $ Syntax $
%   - edges = sladjlist2edgeset(targets, sch)
%
% $ Arguments $
%   - targets:      The targets in adj list (length-n cell array)
%   - sch:          The scheme of conversion
%                   - 0: no value -> no value
%                   - 1: no value -> has value
%                   - 2: has value -> no value
%                   - 3: has value -> has value
%   - edges:        The edge set (nedges x 2 or nedges x 3 matrix)
% 
% $ Remarks $
%   - an internal function for graph representation conversion.
%     no checking of input arguments would be performed.
%
% $ History $
%   - Created by Dahua Lin, on Sep 9, 2006
%

%% get J and V

n = length(targets);
jv = vertcat(targets{:});

if ~isempty(jv)
    if sch == 1
        nedges = size(jv, 1);
        jv = [jv, ones(nedges, 1)];
    elseif sch == 2
        jv = jv(:,1);
    end
else
    edges =[];
    return;
end
        
%% add I

nums = zeros(n, 1);
for i = 1 : n
    cinds = targets{i};
    if ~isempty(cinds)
        nums(i) = size(cinds, 1);
    end
end
ic = slexpand(nums);
edges = [ic, jv];


