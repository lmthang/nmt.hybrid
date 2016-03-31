function Y = sladdvec(X, v, d)
%SLADDVEC adds a vector to columns or rows of a matrix
%
% $ Syntax $
%   - Y = sladdvec(X, v, d)
%   - Y = sladdvec(X, v)
%
% $ Arguments $
%   - X:        The original matrix
%   - v:        The addend vector
%   - d:        The dimension along which the vector is to add
%   - Y:        The resultant matrix
%
% $ Description $
%   - Y = sladdvec(X, v, d) selects the most efficienct way to add a 
%     vector v to every column/row of X. If d == 1, then v should be 
%     a column vector, and is added to each column of X, if d == 2,
%     then v should be a row vector, and is added to each row of X.
%
%   - Y = sladdvec(X, v) will automatically determine d according to
%     the shape of v.
%
% $ Remarks $
%   - The implementation simply wraps the mex function vecop_core.
%
% $ History $
%   - Created by Dahua Lin, on Sep 10, 2006
%

if nargin < 3
    if size(v, 2) == 1
        d = 1;
    else
        d = 2;
    end
end

Y = vecop_core(X, v, d, 1);  % 1 is the opcode of addition in vecop_core







    

