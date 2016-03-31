function Ar = sladd(A0, As, d)
%SLADD Add a sub-array along some dimensions to an array
%
% $ Syntax $
%   - Ar = sladd(A0, As)
%   - Ar = sladd(A0, As, d)
%
% $ Arguments $
%   - A0:           the original array
%   - v:            the sub-array to be added to the array
%   - Ar:           the resultant array
%   - d:            the dimension along which the vector is added
%
% $ Description $
%   - Ar = sladd(A0, As) adds the sub-array As to the array A0 along 
%     auto-selected dimensions. The dimensions are identified by the 
%     dimension of As with size larger than 1. If As is a scalar, then 
%     all elements of A0 will be added As.
%   
%   - Ar = sladd(A0, As, d) adds the sub-array As to the array A0 along
%     the dimensions specified by d. 
%
% $ Remarks $
%   # An empty As is allowed. In such case, the original array A0 will
%     be output, i.e. Ar = A0.
%   # The sizes of dimensions along which the sub-array is added should
%     match that of A0, otherwise, an error will be raised.
%   # By specifying the dimensions through d, the speed can be accelerated.
%
% $ Examples $
%   - Add a vector to a matrix.
%     \{
%         A = [1 2 3; 4 5 6];
%         v = [2; 5];
%         Ar = sladd(A, v)
%     
%         Ar = 
%
%             3     4     5         
%             9    10    11
%
%     \}
%     It is equivalent to sladd(A, v, 1).
%
%  - Add a plane to a matrix
%    \{
%        A1 = [1 2 3; 4 5 6];
%        A2 = [7 8 9; 10 11 12];
%        A = cat(3, A1, A2);
%        v1 = [10; 20];
%        v2 = [30; 40];
%        As = cat(3, v1, v2)
%
%        Ar(:, :, 1) = 
%            
%            11    12    13
%            24    25    26
%
%        Ar(:, :, 2) = 
%
%            37    38    39
%            50    51    52
%
%    \}
%
% $ History $
%   - Created by Dahua Lin on Nov 18th, 2005
%

%% parse and verify input
if nargin < 2
    raise_lackinput('sladd', 2);
end
if isempty(As)
    Ar = A0;
    return;
end
if ndims(As) > ndims(A0)
    error('sltoolbox:dimoverflow', ...
        'The dimension of As should not be larger than that of A0');
end
if nargin < 3 || isempty(d)
    % d is not specified, automatically determine d
    d = find(size(As) > 1);
end
siz_A0 = size(A0);
siz_As = size(As);
if ~isequal(siz_A0(d), siz_As(d))
    error('sltoolbox:dimmismatch', ...
        'The dimensions of As does not match that of A0 in the dimensions to be added');
end

%% compute
siz_A0(d) = 1;
Ar = A0 + repmat(As, siz_A0);

    



