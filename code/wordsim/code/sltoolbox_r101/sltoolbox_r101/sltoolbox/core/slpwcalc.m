function M = slpwcalc(v1, v2, op)
%SLPWCALC Calculates a table pairwisely on two set of scalars
%
% $ Syntax $
%   - M = slpwcalc(v1, v2, op)
%
% $ Arguments $
%   - v1:       The vector of the first set of numbers (length n1)     
%   - v2:       The vector of the second set of numbers (length n2)
%   - op:       The calculation type
%   - M:        The calculated table (n1 x n2)
%
% $ Description $
%   - M = slpwcalc(v1, v2, op) calculates theon the scalars in v1 and v2
%     pairwisely to make the result table M. op is the string indicating
%     the calculation type to take. Available op include:
%       'add':      addition:       M(i, j) = v1(i) + v2(j)
%       'mul':      multiplication: M(i, j) = v1(i) * v2(j)
%       'absdiff':  absolute diff:  M(i, j) = abs(v1(i) - v2(j))
%       'max':      maximum value:  M(i, j) = max(v1(i), v2(j))
%       'min':      minimum value:  M(i, j) = min(v1(i), v2(j))
%
% $ Remarks $
%   - It simply wraps the core mex: pwcalc_core
%
%   - both v1 and v2 should be vectors
% 
% $ History $
%   - Created by Dahua Lin, on Sep 11st, 2006
%

%% parse and verify input

if nargin < 3
    raise_lackinput('slpwcalc', 3);
end

if ~isvector(v1) || ~isvector(v2)
    error('sltoolbox:invalidarg', 'both v1 and v2 should be vectors');
end

switch op
    case 'add'
        opcode = 1;
    case 'mul'
        opcode = 2;
    case 'absdiff'
        opcode = 3;
    case 'max'
        opcode = 4;
    case 'min'
        opcode = 5;
    otherwise
        error('sltoolbox:invalidarg', 'Invalid op type for pwcalc: %s', op);
end

%% main

M = pwcalc_core(v1, v2, opcode);
        






