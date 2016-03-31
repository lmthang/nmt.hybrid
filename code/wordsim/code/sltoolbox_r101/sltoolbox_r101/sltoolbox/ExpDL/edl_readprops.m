function S = edl_readprops(filename, nodetag)
%EDL_READPROPS Reads properties from a property table XML file
%
% $ Syntax $
%   - S = edl_readprops(filename, nodetag)
%
% $ Arguments $
%   - filename:     the filename of the XML file describing the properties
%   - nodetag:      the tag name of each child node of the document node
%   - S:            the struct array of read document
%                   - 'tag':  the document node tag
%                   - 'attribs': the document node attributes
%                   - the struct array using nodetag as fieldname
%
% $ Description $
%   - S = edl_readprops(filename, nodetag) reads a table of properties from 
%     a property table XML file.
%
% $ History $
%   - Created by Dahua Lin, on Aug 10th, 2006
%   - Modified by Dahua Lin, on Aug 13rd, 2006
%       - adds support of document header attribute
%       - changes the structure of the result
%       - adds the selection of node tag
%

%% Read file

if nargin < 2
    raise_lackinput('edl_readprops', 2);
end
    
xdoc = xmlread(filename);
docelem = xdoc.getDocumentElement;

%% Read header

S.tag = char(docelem.getTagName);
S.attribs = xml_getattribs(docelem);


%% Read properties

propElemList = docelem.getElementsByTagName(nodetag);
n = propElemList.getLength;

if n > 0
    % pre-allocation
    % (this can remarkably accelerates the construction of large struct array)
    attrMap = propElemList.item(0).getAttributes;
    firstattr = attrMap.item(0);
    firstname = char(firstattr.getName);
    entries = struct(firstname, cell(n, 1));

    for i = 1 : n

        curentry = propElemList.item(i-1);
        attrMap = curentry.getAttributes;
        na = attrMap.getLength;

        for j = 1 : na
            attr = attrMap.item(j-1);
            attrname = char(attr.getName);
            attrval = char(attr.getValue);
            entries(i).(attrname) = attrval;
        end

    end
else    
    entries = [];
end

S.(nodetag) = entries;



