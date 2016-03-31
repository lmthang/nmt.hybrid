function slmetric_pw_blks(X1, X2, ps, dstpath, mtype, varargin)
%SLMETRIC_PW_BLKS Compute the pairwise metrics in a blockwise manner
%
% $ Syntax $
%   - slmetric_pw_blks(X1, X2, ps, dstpath, mtype, ...)
%
% $ Arguments $
%   - X1:       the first sample matrix
%   - X2:       the second sample matrix
%   - ps:       the partition structure of the score matrix
%   - dstpath:  the destination path
%   - mtype:    the metric type
%
% $ Description $
%   - slmetric_pw_blks(X1, X2, ps, dstpath, mtype, ...) computes the 
%     pairwise metric values for large dataset in blockwise manner.
%     Each block of scores are computed respectively and stored in files.
%     This function is an extension of slmetric_pw to support 
%     blockwise score computation. 
%     Please refer to slmetric_pw and slpwcomp_blks for more information
%     on the usage.
%
% $ History $
%   - Created by Dahua Lin, on Aug 9th, 2006
%

if nargin < 5
    raise_lackinput('slmetric_pw_blks', 5);
end

slpwcomp_blks(X1, X2, ps, dstpath, 'slmetric_pw', mtype, varargin{:});
