function d = sltensor_dot(T1, T2)
%SLTENSOR_DOT Computes the dot product between two tensors
%
% $ Syntax $
%   - d = sltensor_dot(T1, T2)
%
% $ Description $
%   - d = sltensor_dot(T1, T2) Computes the dot product between two tensors
%   T1 and T2.
%
% $ History $
%   - Created by Dahua Lin on June 6th, 2005
%

d = sum(T1(:) .* T2(:));
