function edl_writeprops(doctag, attribs, nodeTag, props, filename)
%EDL_WRITEPROPS Writes the property table to XML file
%
% $ Syntax $
%   - edl_writeprops(doctag, attribs, nodeTag, props, filename)
%
% $ Arguments $
%   - doctag:       the tag name of document node
%   - attribs:      the header attributes
%   - nodeTag:      the tag name of child nodes
%   - props:        the struct array of all properties
%   - filename:     the name of the XML file to be written to
%
% $ Description $
%   - edl_writeprops(doctag, attribs, nodeTag, props, filename) writes 
%     a table of properties to an XML property table file, with the
%     header attributes.
%
% $ Remarks $
%   - For numeric values, they will be converted to string using num2str.
%
% $ History $
%   - Created by Dahua Lin, on Aug 10th, 2006
%   - Modified by Dahua Lin, on Aug 13rd, 2006
%       - adds support of document header attribute
%       - changes the structure of the result
%       - adds the selection of node tag

import com.mathworks.xml.*;

%% parse and verify input arguments

if nargin < 5
    raise_lackinput('edl_writeprops', 5);
end

if ~isstruct(props)
    error('sltoolbox:invalidarg', ...
        'props should be a struct array');
end

%% Write file

% create document

xdoc = XMLUtils.createDocument(doctag);
xdoc.setVersion('1.0');
xdoc.setEncoding('UTF-8');

docelem = xdoc.getDocumentElement;


% write header
docelem = xml_addattribs(docelem, attribs);


% add elements

n = numel(props);
fns = fieldnames(props);
cf = length(fns);

for i = 1 : n
    
    propelem = xdoc.createElement(nodeTag);
    cur = props(i);
    
    for j = 1 : cf
        fn = fns{j};
        val = cur.(fn);
        if isnumeric(val)
            val = num2str(val);
        end
        propelem.setAttribute(fn, val);
    end
    
    docelem.appendChild(propelem);
    
end

% write to file
filedir = slfilepart(filename, 'parent');
slmkdir(filedir);
xmlwrite(filename, xdoc);

    
    
    