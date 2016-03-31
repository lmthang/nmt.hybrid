function b = slisabspath(pstr)
%SLISABSPATH Judges whether the path string is a absolute path
%
% $ Syntax $
%   - b = slisabspath(pstr)
%
% $ Arguments $
%   - pstr:     the path string
%   - b:        a bool variable indicating whether it is a full path
%
% $ History $
%   - Created by Dahua Lin, on Aug 13, 2006
%

b = (length(pstr) >= 2 && ...
        (pstr(2) == ':' || pstr(1) == '\' || pstr(1) == '/'));