function DS = readfile(DS, filename)
%READFILE Reads the dataset from DSDML file
%
% $ Syntax $
%   - DS = readfile(DS, filename) 
%
% $ Arguments $
%   - DS:           the dataset to be loaded from file
%   - filename:     the DSDML file describing the dataset.
%
% $ Description $
%   - DS = readfile(DS, filename) reads the dataset from the specified
%     DSDML file and returns it.
%
% $ History $
%   - Created by Dahua Lin on Jul 23rd, 2006
%


%% Read file

xdoc = xmlread(filename);
docelem = getDocumentElement(xdoc);

%% Parse Header
classname = class(DS);
S = struct(DS);

S.version = char(docelem.getAttribute('version'));
S.name = char(docelem.getAttribute('name'));
S.unittype = char(docelem.getAttribute('unit'));
S.format = char(docelem.getAttribute('format'));
if (docelem.hasAttribute('author'))
    S.author = char(docelem.getAttribute('author'));
else
    S.author = [];
end
if (docelem.hasAttribute('description'))
    S.description = char(docelem.getAttribute('description'));
else
    S.description = [];
end
S.attribs = xml_getattribs(docelem, 'exclude', {...
    'version', ...
    'name', ...
    'unit', ...
    'format', ...
    'author', ...
    'description' ...
    'attribs'});

if ~ismember(S.unittype, {'Sample', 'SampleGroup'})
    error('dsdml:invalidutype', ...
        'Invalid unit type %s', S.unittype);
end


%% Get Units
S.units = [];
unitnodes = docelem.getElementsByTagName(S.unittype);
numnodes = unitnodes.getLength;

ucells = cell(numnodes, 1);

switch(S.unittype)
    case 'Sample'
        for i = 1 : numnodes
            ucells{i} = CreateSample(unitnodes.item(i-1));
        end
        
    case 'SampleGroup'
        for i = 1 : numnodes
            ucells{i} = CreateSampleGroup(unitnodes.item(i-1));
        end
end
S.units = [ucells{:}];
S.units = S.units(:);

DS = class(S, classname);



%%  Specific Functions

function S = CreateSample(node)

S.class_id = str2double(char(node.getAttribute('class_id')));
S.filename = [];
if node.hasAttribute('filename')
    S.filename = char(node.getAttribute('filename'));
end
S.attribs = xml_getattribs(node, 'exclude', {'class_id', 'filename', 'attribs'});


function S = CreateSampleGroup(node)

S.class_id = str2double(char(node.getAttribute('class_id')));
S.attribs = xml_getattribs(node, 'exclude', {'class_id', 'attribs'});
S.samples = [];

samplenodes = node.getElementsByTagName('Sample');
numsamples = samplenodes.getLength;
scells = cell(numsamples, 1);
for i = 1 : numsamples
    scells{i} = CreateSample(samplenodes.item(i-1));
end

S.samples = [scells{:}];
S.samples = S.samples(:);











