function A = slreadarray(filename)
%SLREADARRAY Reads an array from an array file
%
% $ Syntax $
%   - A = slreadarray(filename)
%
% $ Arguments $
%   - filename:         the filename of the array file
%   - A:                the array read
%
% $ Description $
%   - A = slreadarray(filename) reads an array from an array file and
%     returns it by A.
%
% $ History $
%   - Created by Dahua Lin, on Jul 26th, 2006
%

value_types = { ...
    'double', ...
    'single', ...
    'logical', ...
    'char', ...
    'int8', ...
    'uint8', ...
    'int16', ...
    'uint16', ...
    'int32', ...
    'uint32', ...
    'int64', ...
    'uint64'};

%% open file
fid = fopen(filename, 'r');
if fid <= 0
    error('sltoolbox:filefail', ...
        'Fail to open file %s', filename);
end


%% read header

% read and verify tag
tag = fread(fid, 4, '*char')';
if ~isequal(tag, ['arr', 0])
    error('sltoolbox:parseerror', ...
        'The file tag is invalid');
end

% read the value type
typeidx = fread(fid, 1, 'uint8');
valtype = value_types{typeidx};

% read the dimension
d = fread(fid, 1, 'uint8');

%% read size

fseek(fid, 8, -1);
siz = fread(fid, d, 'uint32')';


%% read data
A = fread(fid, prod(siz), ['*', valtype]);
if length(siz) == 1
    siz = [siz, 1];
end
A = reshape(A, siz);


%% close file
fclose(fid);










    