function nrm = sltensor_norm(T)
%SLTENSOR_NORM Computes the Frobenius norm of a tensor T
%
% $ Syntax $
%   - nrm = sltensor_norm(T)
%
% $ Arguments $
%   - T:        the tensor
%   - nrm:      the Frobenius norm of the tensor
%
% $ Description $
%   - nrm = sltensor_norm(T) Computes the Frobenius norm of a tensor T
%
% Copyright Dahua Lin, The MMLab, CUHK
% $Date: 2006/08/13 04:42:42 $
%

nrm = sqrt(sum(T(:).* T(:)));