function Ac = slconfine(A, lb, ub)
%SLCONFINE Confines the values in a range
%
% $ Syntax $
%   - Ac = slconfine(A, lb) 
%   - Ac = slconfine(A, lb, [])
%   - Ac = slconfine(A, [], ub)
%   - Ac = slconfine(A, lb, ub)
%
% $ Arguments $
%   - A:        the input array
%   - lb:       the lower bound to be enforced
%   - ub:       the upper bound to be enforced
%   - Ac:       the confined array
%
% $ Description $
%   - Ac = slconfine(A, lb) confines the values in A to be no less than the 
%          lower bound lb. It is equivalent to Ac = slconfine(A, lb, []).
%
%   - Ac = slconfine(A, [], ub) confines the values in A to be no much than
%          the higher bound ub. 
%
%   - Ac = slconfine(A, lb, ub) confines the values in A to be within the
%          range between lb and ub.
%
% $ Remarks $
%   # The lb and ub can be either a scalar or an array of the same size as
%     the input matrix A.
%
% $ History $
%   - Created by Dahua Lin on Nov 18th, 2005
%

%% parse and verify input arguments
if nargin < 2
    raise_lackinput('slconfine', 2);
end
if nargin < 3
    ub = [];
end

%% compute
Ac = A;
if ~isempty(lb)
    Ac = max(Ac, lb);
end
if ~isempty(ub)
    Ac = min(Ac, ub);
end


