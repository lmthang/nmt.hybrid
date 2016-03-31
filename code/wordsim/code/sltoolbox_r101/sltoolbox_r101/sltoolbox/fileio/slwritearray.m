function slwritearray(A, filename)
%SLWRITEARRAY Writes an array to an array file
%
% $ Syntax $
%   - slwritearray(A, filename)
%   
% $ Arguments $
%   - A:            The array to be written
%   - filename:     The filename of the array file
%
% $ Description $
%   - slwritearray(A, filename) writes an array A to the array file.
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

fid = fopen(filename, 'w');
if fid <= 0
    error('sltoolbox:filefail', ...
        'Fail to open file %s', filename);
end

%% write header

% write tag
fwrite(fid, ['arr', 0], 'char');

% write value type and dimension number
[tf, typeidx] = ismember(class(A), value_types);
if ~tf
    error('Unknown type for A: %s', class(A));
end
d = ndims(A);
info = uint8([typeidx, d, 0, 0]);
fwrite(fid, info, 'uint8');

%% write size

siz = size(A);
fwrite(fid, uint32(siz), 'uint32');

%% write data

fwrite(fid, A, class(A));

%% close file

fclose(fid);






