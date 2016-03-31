function Inds = sllabelinds(labels, labelset)
%SLLABELINDS Extract indices corresponding to specified labels
%
% $ Syntax $
%   - Inds = sllabelinds(labels, labelset)
%
% $ Arguments $
%   - labels:       The labels of samples
%   - labelset:     The set of labels whose indices to be extracted
%   - Inds:         The cell array of indices extracted for labelset
%
% $ Description $
%   - Inds = sllabelinds(labels, labelset) extracts the indices 
%     corresponding to the labels specified in labelset. Suppose the
%     labelset is given by [l1, l2, ...], then Inds would be like
%     {[i11, i12, ...], [i21, i22, ...], ...}, where [i11, i12, ...] is
%     a row vector of indices corresponding to l1, so that 
%     labels(i11) = labels(i12) = ... = l1.
%   
% $ History $
%   - Created by Dahua Lin, on Aug 31, 2006
%

%% parse and verify input

if ~isvector(labels) || ~isnumeric(labels)
    error('sltoolbox:invalidarg', ...
        'labels should be a numeric vector');
end

if size(labels, 1) ~= 1
    labels = labels(:)';
end

%% re-arrange

[labels, si] = sort(labels, 2, 'ascend');
[nums, labelfound] = slcount(labels);
[sinds, einds] = slnums2bounds(nums);
[sfound, smap] = ismember(labelset, labelfound);

%% extract

c = length(labelset);
Inds = cell(1, c);
for i = 1 : c
    if sfound(i)
        mi = smap(i);
        curinds = si(sinds(mi):einds(mi));
        Inds{i} = curinds;
    else
        Inds{i} = [];
    end
end







