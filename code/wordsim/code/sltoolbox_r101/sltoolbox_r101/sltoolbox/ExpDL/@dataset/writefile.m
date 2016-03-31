function writefile(DS, filename)
%WRITEFILE Writes a dataset to a DSDML file
%
% $ Syntax $
%   - writefile(DS, filename)
%
% $ Arguments $
%   - DS:           the dataset object
%   - filename:     the name of the file to be written to
%
% $ Description $
%   - writefile(DS, filename) writes a dataset object to the DSDML file
%     with the filename specified.
%
% $ History $
%   - Created by Dahua Lin on Jul 23rd, 2006
%

import com.mathworks.xml.*;

%% Initialize the document

xdoc = XMLUtils.createDocument('DataSet');
xdoc.setVersion('1.0');
xdoc.setEncoding('UTF-8');

S = struct(DS);


%% Set headers

docelem = xdoc.getDocumentElement;
docelem.setAttribute('version', char(S.version));
docelem.setAttribute('name', char(S.name));
docelem.setAttribute('unit', char(S.unittype));
docelem.setAttribute('format', char(S.format));
if ~isempty(S.author)
    docelem.setAttribute('author', char(S.author));
end
if ~isempty(S.description)
    docelem.setAttribute('description', char(S.description));
end
if ~isempty(S.attribs)
    attrnames = fieldnames(S.attribs);
    nattrs = length(attrnames);
    for i = 1 : nattrs
        docelem.setAttribute(attrnames{i}, S.attribs.(attrnames{i}));
    end
end


%% Create unit nodes

switch S.unittype
    case 'Sample'
        n = length(S.units);
        for i = 1 : n
            docelem.appendChild(CreateSampleNode(xdoc, S.units(i)));
        end
        
    case 'SampleGroup'
        n = length(S.units);
        for i = 1 : n
            docelem.appendChild(CreateSampleGroupNode(xdoc, S.units(i)));
        end
        
    otherwise
        error('dsdml:invalidutype', ...
            'Invalid unit type %s', DS.unittype);
end
    

%% Write document
xmlwrite(filename, xdoc);



%% Node Creating functions

function sampleNode = CreateSampleNode(xdoc, sample)

sampleNode = xdoc.createElement('Sample');
sampleNode.setAttribute('class_id', int2str(sample.class_id));
if ~isempty(sample.filename)
    sampleNode.setAttribute('filename', sample.filename);
end
if ~isempty(sample.attribs)
    attrnames = fieldnames(sample.attribs);
    nattrs = length(attrnames);
    for i = 1 : nattrs
        sampleNode.setAttribute(attrnames{i}, sample.attribs.(attrnames{i}));
    end
end


function groupNode = CreateSampleGroupNode(xdoc, grp)

groupNode = xdoc.createElement('SampleGroup');
groupNode.setAttribute('class_id', int2str(grp.class_id));
if ~isempty(grp.attribs)
    attrnames = fieldnames(grp.attribs);
    nattrs = length(attrnames);
    for i = 1 : nattrs
        sampleNode.setAttribute(attrnames{i}, grp.attribs.(attrnames{i}));
    end
end
nsamples = length(grp.samples);
for i = 1 : nsamples
    groupNode.appendChild(CreateSampleNode(xdoc, grp.samples(i)));
end






