function IMs = slinv2x2(Ms)
%SLINV2X2 Computes inverse matrices for 2 x 2 matrices in a fast way
%
% $ Syntax $
%   - IMs = slinv2x2(Ms)
%
% $ Arguments $
%   - Ms:       the 2 x 2 matrix (matrices)
%   - IMs:      the computed inverse matrix (matrices)
%
% $ Description $
%   - IMs = slinv2x2(Ms) computes the inverse of 2x2 matrices. If Ms is
%     a 2 x 2 matrix, then r is its inverse.
%     Or, if Ms is a 2 x 2 x ... array, then r would be a 2 x 2 x ... array 
%     storing the inverse matrices.
%
% $ Remarks $
%   - The function uses the following formula for fast calculation of
%     inverse of 2 x 2 matrices:
%       inv(A) = [a22, -a12; -a21, a11] / det(A)
%
% $ History $
%   - Created by Dahua Lin on Apr 22nd, 2006
%

%% parse and verify input arguments

if size(Ms, 1) ~= 2 || size(Ms, 2) ~= 2
    error('sltoolbox:invalidarg', 'Ms should be set of 2 x 2 matrices');
end

%% compute

if ndims(Ms) == 2       % single matrix
    IMs = [Ms(4), -Ms(2); -Ms(3), Ms(1)] / (Ms(1) * Ms(4) - Ms(2) * Ms(3));
    
else                    % a set of matrices
    
    IMs = zeros(size(Ms));
    
    % compute adjacent matrix
    IMs(1,1,:) = Ms(2,2,:);
    IMs(1,2,:) = -Ms(1,2,:);
    IMs(2,1,:) = -Ms(2,1,:);
    IMs(2,2,:) = Ms(1,1,:);
    
    % compute the determinants
    d = sldet2x2(Ms);
    
    % scale
    d = reshape(d, [1, 1, size(d)]);
    IMs = slmul(IMs, 1 ./ d);
        
end

