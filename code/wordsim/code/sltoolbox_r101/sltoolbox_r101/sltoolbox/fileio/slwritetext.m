function slwritetext(T, filename)
%SLWRITETEXT Writes a text to file
%
% $ Syntax $
%   - slwritetext(T, filename)
%
% $ Arguments $
%   - T             the cell array representing the text
%   - filename      the path of the file to write to
%
% $ Description $
%   - slwritetext(T, filename) Writes in a text file stored in a cell array 
%   to a text file specified by filename.
%
% $ History $
%   Created by Dahua Lin, on 2005-06-02
%

if ~iscell(T)
    error('T should be a text cell array');
end
fid = fopen(filename, 'w');
if (fid <= 0)
    error(['Fail to open ', filename]);
end

n = length(T);
for i = 1 : n
    fprintf(fid, '%s\r\n', T{i});
end
    
fclose(fid);