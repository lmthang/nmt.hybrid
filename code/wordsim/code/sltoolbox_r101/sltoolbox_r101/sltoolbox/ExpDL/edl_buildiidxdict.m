function Dict = edl_buildiidxdict(S, idxfn)
%EDL_BUILDIIDXDICT Builds a dictionary using internal index
%
% $ Syntax $
%   - Dict = edl_buildiidxdict(S, idxfn)
%
% $ Arguments $
%   - S:        a struct array on which the dictionary is built
%   - idxfn:    the field name of the internal index 
%               (default = 'internal_index')
%   - Dict:     the built dictionary
%               a cell array, indexed by internal_index
%
% $ Remarks $
%   - The length of the cell array (dict) will be equal to the maximum
%     internal index in S.
%   - In S, different elements should have different internal indices.
%   - For the index corresponding to no element, the corresponding cells
%     are empty.
%   - The dictionary will not automatically keep track of the change of
%     S. If S is changed, the dictonary should be rebuilt in order to
%     reflect the updates.
%
% $ History $
%   - Created by Dahua Lin, on Aug 15, 2006
%

%% parse and verify input arguments

if ~isstruct(S)
    error('sltoolbox:invalidarg', ...
        'The S should be a struct');
end

if nargin < 2 || isempty(idxfn)
    idxfn = 'internal_index';
end

if ~isfield(S, idxfn)
    error('sltoolbox:invalidarg', ...
        'The S have no specified internal index field');
end

%% Build dictionary

% gather the internal indices
n = numel(S);
inds = zeros(n, 1);
for i = 1 : n
    idx = S(i).(idxfn);
    if ischar(idx)
        idx = str2double(idx);
    end
    inds(i) = idx;
end

% construct dictionary
m = max(inds);

Dict = cell(m, 1);

for i = 1 : n
    idx = inds(i);
    if ~isempty(Dict{idx})
        error('sltoolbox:rterror', ...
            'Repeated index %d is encountered', idx);
    end
    Dict{idx} = S(i);        
end

