function [name, value] = slparse_assignment(str)
%SLPARSE_ASSIGNMENT Parses an assignment string
%
% $ Syntax $
%   - [name, value] = slparse_assignment(str)
%
% $ Arguments $
%   - str:          the string representing an assignment
%   - name:         the name of the value
%   - value:        the value
%
% $ Description $
%   - [name, value] = slparse_assignment(str) parses an assignment string
%     and extracts the name and value. The string should be in the form
%     of <name> = <value>. For the syntax of assignment, we have following
%     rules:
%       - spaces are allowed between <name> and = and between = and <value>
%       - <name> should be a valid Matlab name, which should be checked
%         by isvarname
%       - The form of <value> can be either of the following:
%           - a numeric scalar (will be converted to double)
%           - a matrix as expressed in matlab (will be converted to matrix)
%           - a string (will be converted to char string)
%           - a string quoted by ' or " (will be de-quoted)
%
% $ Remarks $
%   - The whole string can have multiple =, but only the first one will
%     be considered as the assignment mark.
%
% $ History $
%   - Created by Dahua Lin, on Aug 9th, 2006
%

%% parse and verify input arguments

if ~ischar(str)
    error('sltoolbox:invalidarg', ...
        'The str should be a char string');
end

str = strtrim(str);
name = [];
value = [];
if isempty(str)
    return;
end


%% divide parts

peq = find(str == '=', 1);
if isempty(peq)
    error('sltoolbox:parseerror', ...
        'Fail to locate the assignment equal mark');
end

name = strtrim(str(1:peq-1));
value = strtrim(str(peq+1:end));

%% check name

if isempty(name) || ~isvarname(name)
    error('sltoolbox:parseerror', ...
        'The name is invalid for assignment in %s', str);
end

%% process value

if isempty(value)
    value = [];
    return;
end

trynums = str2num(value);
if ~isempty(trynums)
    value = trynums;
elseif length(value) >= 2 && ...
        ((value(1) == '''' && value(end) == '''') || (value(1) == '"' && value(end) == '"'))
    value = value(2:end-1);
    if isempty(value)
        value = '';
    end
end

