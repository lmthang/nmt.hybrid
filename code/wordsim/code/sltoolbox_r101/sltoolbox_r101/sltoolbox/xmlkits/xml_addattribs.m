function xelem = xml_addattribs(xelem, varargin)
%XML_ADDATTRIBS Adds attributes to an element
%
% $ Syntax $
%   - xelem = xml_addattribs(xelem, attrname1, attrvalue1, ...)
%   - xelem = xml_addattribs(xelem, S)
%
% $ Description $
%   - xelem = xml_addattribs(xelem, attrname1, attrvalue1, ...) adds 
%     the attributes specified in the property list to the XML element.
%
%   - xelem = xml_addattribs(xelem, S) adds the attributes specified in
%     the struct S (the field names are the attribute names, while the 
%     field values are the attribute values).
%
% $ Remarks $
%   - If the attributes have existed in the XML element, the attribute
%     values will be overwritten.
%   - The input element will be changed in the function. Please use
%     the same variable for the XML element in both input and output.
%
% $ History $
%   - Created by Dahua Lin on Jul 23rd, 2006
%   - Modified by Dahua Lin on Aug 13rd, 2006
%

%% Parse the attributes to be added to the 

if isempty(varargin)
    return;
end

if isempty(varargin{1})
    return;
elseif ischar(varargin{1})
    nvars = length(varargin);
    if mod(nvars, 2) ~= 0
        error('The number of variables in the property list should be even');
    end
    nattrs = nvars / 2;
    attrnames = varargin(1:2:end)';
    attrvals = varargin(2:2:end)';
elseif isstruct(varargin{1})
    S = varargin{1};
    attrnames = fieldnames(S);
    nattrs = length(attrnames);
    attrvals = cell(nattrs, 1);
    for i = 1 : nattrs
        attrvals{i} = S.(attrnames{i});
        if ~ischar(attrvals{i})
            error('The attribute values should be char arrays');
        end
    end
else
    error('The attributes should be specified by either property list or a struct');
end

%% Add attributes
for i = 1 : nattrs
    setAttribute(xelem, attrnames{i}, attrvals{i});
end
    
    
    
    