function R = get(DS, props)
%GET gets the properties of the dataset
%
% $ Syntax $
%   - R = get(DS, propname)
%   - R = get(DS, propname cell array)
%
% $ Description $   
%   - R = get(DS, propname) gets a property value with its name specified
%     by propname
%
%   - R = get(DS, propname cell array) gets a set of properties with the
%     names given in a cell array. Consequently, the property values are
%     returns through a cell array of the same size.
%
% $ Remarks $
%   - The 
% $ History $
%   - Created by Dahua Lin on Jul 23rd, 2006
%

%% Skeleton

if ischar(props)    
    S = struct(DS);
    R = getprop(S, props);
    
elseif iscell(props)
    S = struct(DS);
    R = cell(size(props));
    n = numel(R);
    for i = 1 : n
        if ~ischar(props{i})
            error('dsdml:invalidarg', ...
                'The property name should be a char string');
        end
        R{i} = getprop(S, props{i});
    end
    
else
    error('dsdml:invalidarg', ...
        'The props should be a property name or a cell of property names');
end


%% Property getting function

function R = getprop(S, propname)

if isfield(S, propname)
    R = S.(propname);
elseif ~isempty(S.attribs) && isfield(S.attribs, propname)
    R = S.attribs.(propname);
else
    switch(propname)
        case 'numsamples'
            R = getnumsamples(S);
        case 'numgroups'
            R = getnumgroups(S);
        case 'numunits'
            R = length(S.units);
            
        case 'samples'
            R = getsamples(S);
        case 'groups'
            R = getgroups(S);
        case 'class_ids'
            R = getclass_ids(S);
        case 'filenames'
            R = getfilenames(S);
        case 'nums'
            R = getnums(S);
            
        case 'numspergrp'
            R = getnumspergrp(S);
        case 'grpnums'
            R = getgrpnums(S);
        case 'grpclass_ids'
            R = getgrpclass_ids(S);
            
        otherwise
            error('dsdml:invalidarg', ...
                'Invalid property name of the dataset: %s', propname);
    end
end


%% Specific property getting functions

function R = getnumsamples(S)

switch S.unittype
    case 'Sample'
        R = length(S.units);
    case 'SampleGroup'
        R = 0;
        ng = length(S.units);
        for i = 1 : ng
            R = R + length(S.units(i).samples);
        end
    otherwise
        invalid_utype(S);
end


function R = getnumgroups(S)

switch S.unittype
    case {'Sample', 'SampleGroup'}
        R = length(S.units);
    otherwise
        invalid_utype(S);
end


function R = getsamples(S)

switch S.unittype
    case 'Sample'
        R = S.units;
    case 'SampleGroup'
        R = vertcat(S.units.samples);
    otherwise
        invalid_utype(S);
end


function R = getgroups(S)

switch S.unittype
    case 'Sample'
        n = length(S.units);
        U = S.units;
        R = struct(...
            'class_id', cell(n, 1), ...
            'samples', cell(n, 1), ...
            'attribs', cell(n, 1));
        for i = 1 : n
            R(i).class_id = U(i).class_id;
            R(i).samples = U(i);
        end
    case 'SampleGroup'
        R = S.units;
    otherwise
        invalid_utype(S);
end


function R = getclass_ids(S)

switch(S.unittype)
    case 'Sample'
        n = length(S.units);
        R = zeros(n, 1);
        for i = 1 : n
            R(i) = S.units(i).class_id;
        end
    case 'SampleGroup'
        n = getnumsamples(S);
        R = zeros(n, 1);
        c = 0;
        ng = length(S.units);
        for i = 1 : ng
            ns = length(S.units(i).samples);
            R(c+1:c+ns) = S.units(i).class_id;
            c = c + ns;
        end
    otherwise
        invalid_utype(S);
end


function R = getfilenames(S)

switch(S.unittype)
    case 'Sample'
        n = length(S.units);
        R = cell(n, 1);
        for i = 1 : n
            R{i} = S.units(i).filename;
        end
    case 'SampleGroup'
        n = getnumsamples(S);
        R = cell(n, 1);
        c = 0;
        ng = length(S.units);
        for i = 1 : ng
            ns = length(S.units(i).samples);
            [R{c+1:c+ns}] = deal(S.units(i).samples.filename);
            c = c + ns;
        end
    otherwise
        invalid_utype(S);
end


function R = getnums(S)

R = countnums(getclass_ids(S));



function R = getnumspergrp(S)

switch (S.unittype)
    case 'Sample'
        n = length(S.units);
        R = ones(n, 1);
        
    case 'SampleGroup'
        n = length(S.units);
        R = zeros(n, 1);
        for i = 1 : n
            R(i) = length(S.units(i).samples);
        end
        
    otherwise
        invalid_utype(S);
end


function R = getgrpnums(S)

R = countnums(getgrpclass_ids(S));
        

function R = getgrpclass_ids(S)

switch (S.unittype)
    case {'Sample', 'SampleGroup'}
        n = length(S.units);
        R = zeros(n, 1);
        for i = 1 : n
            R(i) = S.units(i).class_id;
        end
        
    otherwise
        invalid_utype(S);
end                                    



%% Auxiliary functions

function invalid_utype(S)

error('dsdml:invalidunittype', ...
            'Invalid unit type: %s', S.unittype);


function nums = countnums(A)

A = A(:);
difs = diff(A);
n = length(A);
nums = [0; find(difs ~= 0); n];
nums = diff(nums);





    





