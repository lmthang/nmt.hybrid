function r = sllogdet(A)
%SLLOGDET Computes the logarithm of determinant of a matrix in a robust way
%
% $ Syntax $
%   - r = sllogdet(A)
%
% $ Arguments $
%   - A:       the square matrix whose log-determinant is to be solved
%   - r:       the log-determinant of A
%
% $ Description $
%   - r = sllogdet(A) computes the logarithm of (absolute )determinant of a square
%     matrix A in a robust way, to reduce the risk of underflow or overflow.
%
% $ Remarks $
%   - If the matrix is singular, inf or -inf would be return.
%
% $ History $
%   - Created by Dahua Lin on Dec 28th, 2005
%

%% parse and verify input arguments
if ndims(A) ~= 2
    error('sltoolbox:invalidarg', 'A should be a 2D matrix');
end
d = size(A, 1);
if size(A, 2) ~= d
    error('sltoolbox:invalidarg', 'A shoule be a square matrix');
end

%% compute

% triangular decomposition
[L, U] = lu(A);
diagU = diag(U);
slignorevars(L);
clear L U;

% sign controlling
neg_u_entries = find(diagU < 0);
if ~isempty(neg_u_entries)
    diagU(neg_u_entries) = -diagU(neg_u_entries);
end

% log-product
wid = 'MATLAB:log:logOfZero';
st = warning('query', wid);
st = st.state;

warning('off', wid);
r = sum(log(diagU));
warning(st, wid);




