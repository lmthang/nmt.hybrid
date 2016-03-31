function A = xml_getattribs(xelem, varargin)
%XML_GETATTRIBS Constructs an attribte struct from an XML element
%
% $ Syntax $
%   - A = xml_getattribs(xelem)
%   - A = xml_getattribs(xelem, ...)
%
% $ Arguments $
%   - xelem:          the XML element
%   - A:              the struct of attributes
%                     - the field name represents the attribute name
%                     - the field value (string) is the attribute value
%
% $ Description $
%   - A = xml_getattribs(xelem) constructs the struct representing
%     the attributes in the XML element xelem, with the attribute names
%     stored in the struct fields, while the attribute values stored in the
%     struct values in terms of char string.
%   
%   - A = xml_getattribs(xelem, ...) constructs the
%     attribute struct with following properties:
%       - 'exclude': a cell array of excluded attribute names, default = {}
%       - 'select':  a cell array of selected attribute names, default = {}
%       - 'forceexist': whether the fields must exist for select, 
%                       default = true;
%
% $ Remarks $
%   - If select is specified, then exclude will be ignored.
%   - All selected attributes should exist.
%
% $ History $
%   - Created by Dahua Lin on Jul 23rd, 2006
%


%% Parse and verify input arguments

opts.select = {};
opts.exclude = {};
opts.forceexist = true;
opts = slparseprops(opts, varargin{:});


%% Construction

A = [];



if isempty(opts.select) && isempty(opts.exclude) % full construction

    attrMap = xelem.getAttributes;
    n = attrMap.getLength;

    for i = 1 : n
        
        attr = attrMap.item(i-1);
        attrname = char(attr.getName);
        attrval = char(attr.getValue);
        
        A.(attrname) = attrval;
        
    end
    
    
elseif ~isempty(opts.select)    % selection
    
    n = length(opts.select);
    for i = 1 : n
        
        attrname = opts.select{i};
        if xelem.hasAttribute(attrname)
            A.(attrname) = char(xelem.getAttribute(attrname));
        else
            if opts.forceexist
                error('sltoolbox:xmlerror', ...
                    'The selected attribute %s does not exist', attrname);
            end
        end
        
    end    
    
else                            % has excluding
    
    attrMap = xelem.getAttributes;
    n = attrMap.getLength;

    for i = 1 : n
        
        attr = attrMap.item(i-1);
        attrname = char(attr.getName);
        
        if ~ismember(attrname, opts.exclude)            
            attrval = char(attr.getValue);
            A.(attrname) = attrval;        
        end
        
    end
        
end



    