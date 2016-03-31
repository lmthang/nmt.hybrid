function R = slrotmat(theta)
%SLROTMAT Get the 2x2 rotation matrix
%
% $ Syntax $
%   - R = slrotmat(theta)
%
% $ Arguments $
%   - theta:      the radian of rotation
%   - R:          the rotation matrix
%
% $ Description $
%   - R = slrotmat(theta) computes the 2x2 rotation matrix R which
%     rotates a 2D point anti-clockwisely by radian of theta. 
%     If theta is a scalar, then R will be a 2x2 matrix, if
%     theta is an n1 x n2 x ... array, then R will be a set of 
%     matrices stored in an 2 x 2 x n1 x n2 x ... array.
%
% $ Remarks $
%   - The R is given by following formula:
%     R = [ cos(theta)   -sin(theta);
%           sin(theta)    cos(theta) ];
%
% $ History $
%   - Created by Dahua Lin on Apr 23, 2006
%

if isscalar(theta)  % single
    R = [cos(theta), -sin(theta);
         sin(theta),  cos(theta)];
else % multiple
    TM = reshape(theta, [1, 1, size(theta)]);
    CM = cos(TM);
    SM = sin(TM);
    R = [CM, -SM;
         SM,  CM];
end

    

    