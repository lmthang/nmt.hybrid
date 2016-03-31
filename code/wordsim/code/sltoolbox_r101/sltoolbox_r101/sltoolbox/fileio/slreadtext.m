function T = slreadtext(filename)
%SLREADTEXT Reads in a text file into a cell array
%
% $ Syntax $
%   T = slreadtext(filename)
%
% $ Description $
%   - T = slreadtext(filename) Reads in a text file to a cell array with
%     each cell storing a line string. 
%
% $ Remarks $
%   1. No extra operations will be performed on the strings besides deleting
%      the trailing spaces in end of lines.
%
% $ History $
%   Created by Dahua Lin, on 2005-06-02
%

fid = fopen(filename, 'r');
if (fid <= 0)
    error(['Fail to open file ', filename]);
end

T = {};
icount = 0;
strline = fgets(fid);
while ~isequal(strline, -1)
    icount = icount + 1;
    T{icount, 1} = deblank(strline);
    strline = fgets(fid);
end

fclose(fid);