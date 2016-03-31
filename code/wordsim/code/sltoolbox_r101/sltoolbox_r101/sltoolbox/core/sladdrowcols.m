function Y = sladdrowcols(X, vrow, vcol)
%SLADDROWCOLS Adds the vectors to all rows and all columns
%
% $ Syntax $
%   - Y = sladdrowcols(X, vrow, vcol)
%
% $ Arguments $
%   - X:            The input matrix
%   - vrow:         The row vector to add to all rows
%   - vcol:         The column vector to add to all columns
%   - Y:            The resultant matrix
%
% $ Description $
%   - Y = sladdrowcols(X, vrow, vcol) adds vrow to all rows of X and
%     adds vcol to all columns of Y.
%
% $ Remarks $
%   - The implementation simply wrapps the mex function rowcolop_core.
%
% $ History $
%   - Created by Dahua Lin, on Sep 10, 2006
%

Y = rowcolop_core(X, vrow, vcol, 1); % 1 is the opcode of addition