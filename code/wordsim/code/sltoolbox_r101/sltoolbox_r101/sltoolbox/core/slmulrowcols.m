function Y = slmulrowcols(X, vrow, vcol)
%SLMULROWCOLS Multiplies the vectors to all rows and all columns
%
% $ Syntax $
%   - Y = slmulrowcols(X, vrow, vcol)
%
% $ Arguments $
%   - X:            The input matrix
%   - vrow:         The row vector to multiply to all rows
%   - vcol:         The column vector to multiply to all columns
%   - Y:            The resultant matrix
%
% $ Description $
%   - Y = slmulrowcols(X, vrow, vcol) multiplies vrow to all rows of X and
%     adds vcol to all columns of Y.
%
% $ Remarks $
%   - The implementation simply wrapps the mex function rowcolop_core.
%
% $ History $
%   - Created by Dahua Lin, on Sep 10, 2006
%

Y = rowcolop_core(X, vrow, vcol, 2); % 2 is the opcode of multiplication