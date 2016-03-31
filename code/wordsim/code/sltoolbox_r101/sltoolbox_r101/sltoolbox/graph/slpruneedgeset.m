function edges = slpruneedgeset(n, nt, edges, method)
%SLPRUNEEDGESET Prunes the edge set
%
% $ Syntax $
%   - edges = slpruneedgeset(n, nt, edges)
%   - edges = slpruneedgeset(n, nt, edges, method)
%
% $ Arguments $
%   - n:            The number of (source) nodes
%   - nt:           The number of (target) nodes
%   - edges:        The input/output edge set
%   - method:       The method of merging.
%
% $ Description $
%   - edges = slpruneedgeset(n, nt, edges) prunes the edge set using
%     default method, so that the appositional edges (the edges from the 
%     same source and to the same target) has only one entry in the 
%     edge set. 
%
%   - edges = slpruneedgeset(n, nt, edges, method) prunes the edge set 
%     using the specified method. We have following methods:
%       - 'nomulti':    report an error when multiple entries exist for
%                       the same edge. (for ncols = 2 and 3)
%       - 'noconflict': report an error when multiple entries exist for
%                       the same edge having different values. That multiple
%                       entries exist for the same edge with same values
%                       is allowed. (for ncols = 2 and 3)
%       - 'exist':      Despite the number of ocurrence, 
%                       if ncols = 2, preserve all existent edges
%                       if ncols = 3, set value to all existent edges to 1
%       - 'noval':      preserve all existent edges, for both ncols = 2 
%                       and 3, the value column is discarded. 
%       - 'count':      Set the value to the number of occurrence of 
%                       the edge. (for ncols = 2 and 3, however, the output
%                       always have 3 columns, with the third column being
%                       the count value of the corresponding edge)
%       - 'sum':        Set the value to the sum of the values for the
%                       edges (only for ncols = 3)
%       - 'avg':        Set the value to the average of the values for 
%                       the edges (only for ncols = 3)
%       - 'max':        Set the value to the maximum value for the edges
%                       (only for ncols = 3)
%       - 'min':        Set the value to the minimum value for the edges
%                       (only for ncols = 3)
%       - 'first':      When multiple values exist for the same edge,
%                       only the first one takes effects, others are 
%                       ignored. (for ncols = 2 and 3)
%       - 'last':       When multiple values exist for the same edge,
%                       only the last one takes effects, others are 
%                       ignored. (for ncols = 2 and 3)
%     By default, when ncols == 2, the default method is 'noval', when
%     ncols == 3, the default method is 'last'.
%     The method can also be a function_handle using the following form:
%       V = fp(V0, nums, sinds, einds)
%       The input:
%           - V0: the initial value (may be repeated) groupped by edges
%           - nums: the numbers of values for edges
%           - sinds: the start positions of the values for edges
%           - einds: the end positions of the value for edges
%       It should output V, a number of pruned edges x 1 column vector 
%       with each value for a pruned edge.
%
% $ History $
%   - Created by Dahua Lin, on Sep 9, 2006
%

%% parse and verify input

if nargin < 3
    raise_lackinput('slpruneedgeset', 3);
end    

if isempty(edges)
    return;
end

ncols = size(edges, 2);
if ~isnumeric(edges) || ndims(edges) ~= 2 || (ncols ~= 2 && ncols ~= 3)
    error('sltoolbox:invalidarg', ...
        'The edges should be an nedges x 2 or nedges x 3 matrix');
end

if nargin < 4 || isempty(method)
    if ncols == 2
        method = 'exist';
    else
        method = 'sum';
    end
end

%% delegate

if ischar(method)
    switch method
        case 'nomulti'
            fp = @prune_nomulti;
            require_value = false;
        case 'noconflict'
            fp = @prune_noconflict;
            require_value = false;
        case 'exist'
            fp = @prune_exist;
            require_value = false;
        case 'noval'
            fp = @prune_noval;
            require_value = false;
        case 'count'
            fp = @prune_count;
            require_value = false;
        case 'sum'
            fp = @prune_sum;
            require_value = true;
        case 'avg'
            fp = @prune_avg;
            require_value = true;
        case 'max'
            fp = @prune_max;
            require_value = true;
        case 'min'
            fp = @prune_min;
            require_value = true;
        case 'first'
            fp = @prune_first;
            require_value = false;
        case 'last'
            fp = @prune_last;
            require_value = false;
        otherwise
            error('sltoolbox:invalidarg', ...
                'Unknown edgeset prune method: %s', method);
    end
elseif isa(method, 'function_handle')
    fp = method;
else
    error('sltoolbox:invalidarg', ...
        'The prune method should be either a string or a function handle');
end

if ncols == 2 && require_value
    error('sltoolbox:rterror', ...
        'The prune method requires the value column in the edges');
end

%% arrange edges

% get indices
I = edges(:, 1);
J = edges(:, 2);
inds = sub2ind([n, nt], I, J);

% get the sorting
[inds, si] = sort(inds);
nums = slcount(inds);
clear inds;

% sort I J V0
I = I(si);
J = J(si);
if ncols == 3
    V0 = edges(si, 3);
else
    V0 = [];
end
clear si;


%% do prune on V0

% prune the I J 
[sinds, einds] = slnums2bounds(nums);
I = I(sinds);
J = J(sinds);

% prune the V
V = fp(V0, nums, sinds, einds);

%% merge pruned I J V

if isempty(V)
    edges = [I, J];
else
    edges = [I, J, V];
end


%% value prune functions

function V = prune_nomulti(V0, nums, sinds, einds)

slignorevars(sinds, einds);

if any(nums > 1)
    error('sltoolbox:rterror', ...
        'It is not allowed that multiple entries for the same edge');
end
V = V0;


function V = prune_noconflict(V0, nums, sinds, einds)

slignorevars(einds);

if ~isempty(V0)
    V = V0(sinds);
    if any(nums > 1)
        Vr = slexpand(nums, V);
        if ~isequal(V0, Vr)
            error('sltoolbox:rterror', ...
                'There is confliction between values specified for the same edge');
        end
    end
else
    V = [];
end


function V = prune_exist(V0, nums, sinds, einds)

slignorevars(sinds, einds);

if ~isempty(V0)
    nedges = length(nums);
    V = ones(nedges, 1);
else
    V = [];
end


function V = prune_noval(V0, nums, sinds, einds)

slignorevars(V0, nums, sinds, einds);
V = [];


function V = prune_count(V0, nums, sinds, einds)

slignorevars(V0, sinds, einds);
V = nums;


function V = prune_sum(V0, nums, sinds, einds)

ne = length(nums);
V = zeros(ne, 1);
for i = 1 : ne
    curV0 = V0(sinds(i):einds(i));
    V(i) = sum(curV0);
end


function V = prune_avg(V0, nums, sinds, einds)

V = prune_sum(V0, nums, sinds, einds);
V = V ./ nums;


function V = prune_max(V0, nums, sinds, einds)

ne = length(nums);
V = zeros(ne, 1);
for i = 1 : ne
    curV0 = V0(sinds(i):einds(i));
    V(i) = max(curV0);
end 


function V = prune_min(V0, nums, sinds, einds)

ne = length(nums);
V = zeros(ne, 1);
for i = 1 : ne
    curV0 = V0(sinds(i):einds(i));
    V(i) = min(curV0);
end 


function V = prune_first(V0, nums, sinds, einds)

slignorevars(V0, nums, einds);
if ~isempty(V0)
    V = V0(sinds);
else
    V = [];
end


function V = prune_last(V0, nums, sinds, einds)

slignorevars(V0, nums, sinds);
if ~isempty(V0)
    V = V0(einds);
else
    V = [];
end


