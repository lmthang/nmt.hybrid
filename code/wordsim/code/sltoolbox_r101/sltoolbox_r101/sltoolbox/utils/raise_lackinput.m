function raise_lackinput(funcname, nmin)
%RAISE_LACKINPUT Raises an error indicating lack of input argument
%
% $ Syntax $
%   - raise_lackinput(funcname, nmin)
%
% $ Arguments $
%   - funcname:         the name of invoking function
%   - nmin:             the minimum number of input arguments
%
% $ Description $
%   - raise_lackinput(funcname, nmin) raises an error message indicating
%     that the number of arguments input to the function specified by
%     funcname is not enough. (It should be at least nmin)
%
% $ History $
%   - Created by Dahua Lin on Nov 18th, 2005
%

error('sltoolbox:lackinput', ...
    'The number of input argument to %s should be at least %d.', ...
    funcname, nmin);


