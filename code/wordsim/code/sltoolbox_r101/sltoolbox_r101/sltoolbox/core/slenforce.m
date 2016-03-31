function Ae = slenforce(A, varargin)
%SLENFORCE Enforce some property on the array A
%
% $ Syntax $
%   - Ae = slenforce(A, <property>)
%   - Ae = slenforce(A, <property1>, <property2>, ...)
%   - Ae = slenforce(A, {<property1>, <property2>, ...})
%
% $ Description $
%   - Ae = slenforce(A, <property>) enforces the property to the matrix A.
%
%   - Ae = slenforce(A, <property1>, <property2>, ...) enforces a set of 
%     properties to the matrix A. It is equivalent to 
%     Ae = slenforce(A, {<property1>, <property2>, ...}), where the
%     properties are grouped in a cell form.
%
%   - The properties are listed in following table
%     \*
%     \t  Table 1.  The properties that can be enforced              \\
%     \h    name         &     description                           \\
%          'real'        &     All values are real                   \\
%          'positive'    &     All values are positive               \\
%          'negative'    &     All values are negative               \\
%          'nonpos'      &     All values are non-positive           \\
%          'nonneg'      &     All values are non-negative           \\
%          'symmetric'   &     Matrices are symmetric                \\
%     \*
%
% $ Remarks $
%   # A can be numeric array of any dimensions. For arrays with dimensions
%     higher than 2D, 'symmetric' property is enforced to each page.
%   # slenforce(A, 'positive') is equivalent to slconfine(A, eps, inf); 
%     slenforce(A, 'negative') is equivalent to slconfine(A, -inf, -eps);
%     slenforce(A, 'nonpos') is equivalent to slconfine(A, -inf, 0);
%     slenforce(A, 'nonneg') is equivalent to slconfine(A, 0, inf);
%
% $ History $
%   - Created by Dahua Lin on Nov 18th, 2005
%  

%% parse and verify input arguments
if nargin < 2
    raise_lackinput('slenforce', 2);
end
if ischar(varargin{1})
    P = varargin;
elseif iscell(varargin{1})
    P = varargin{1};
else
    error('sltoolbox:invalidarg', ...
        'The properties should be in strings or cell array of strings');
end
np = numel(P);
for i = 1 : np
    if ~ischar(P{i})
        error('sltoolbox:nonstring', ...
            'Some property names are not given in form of string');
    end
end

%% enforce
Ae = A;
for i = 1 : np
    
    switch P{i}
        case 'real'
            if ~isreal(Ae)
                Ae = real(Ae);
            end
            
        case 'positive'
            Ae = slconfine(Ae, eps, []);
            
        case 'negative'
            Ae = slconfine(Ae, [], -eps);
            
        case 'nonpos'
            Ae = slconfine(Ae, [], 0);
            
        case 'nonneg'
            Ae = slconfine(Ae, 0, []);
            
        case 'symmetric'
            Ae = (Ae + Ae') / 2;
            
        otherwise
            error('sltoolbox:invalidop', ...
                'Unsupported property for slenforce: %s', P{i});
    end
    
end



    

