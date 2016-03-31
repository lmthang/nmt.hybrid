function S = edl_readenvvars(envfile)
%EDL_READENVVARS Reads in a file with environment variables
%
% $ Syntax $
%   - S = edl_readenvvars(envfile)
%
% $ Arguments $
%   - envfile:      the environment filename
%   - S:            the struct of all environment variables
%
% $ Description $
%   - S = edl_readenvvars(envfile) reads in a set of environment variables
%     from an assignment file. The file contains a set of assignment 
%     string, and some comments (starting with # or % or /).
%
% $ History $
%   - Created by Dahua Lin, on Aug 10th, 2006
%

%% read file

T = slreadtext(envfile);
T = slcompresstext(T);

%% parse assignments

n = length(T);
S = [];

for i = 1 : n
    
    curline = T{i};
    if ~iscomment(curline)
        [curname, curval] = slparse_assignment(curline);
        S.(curname) = curval;
    end    
    
end

%% Auxiliary function

function b = iscomment(line)

b = (line(1) == '#' || line(1) == '%' || line(1) == '/');