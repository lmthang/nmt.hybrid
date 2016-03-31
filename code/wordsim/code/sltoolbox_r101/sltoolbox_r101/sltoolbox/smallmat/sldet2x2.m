function r = sldet2x2(Ms)
%SLDET2X2 Computes the determinant of 2 x 2 matrices in a fast way
%
% $ Syntax $
%   - r = sldet2x2(Ms)
%
% $ Arguments $
%   - Ms:       the 2 x 2 matrix (matrices)
%   - r:        the resultant determinant(s)
%
% $ Description $
%   - r = sldet2x2(Ms) computes the determinant of 2x2 matrices. If Ms is
%     a 2 x 2 matrix, then r is a scalar representing its determinant. 
%     Or, if Ms is a 2 x 2 x ... array, then r would be a ... array 
%     storing the determinant of all 2 x 2 matrices.
%
% $ Remarks $
%   - The function uses the following formula for fast calculation of
%     determinant of 2 x 2 matrices:
%       det = a11 * a22 - a12 * a21
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
    r = Ms(1) * Ms(4) - Ms(2) * Ms(3);
    
else                    % a set of matrices
    r = Ms(1,1,:) .* Ms(2,2,:) - Ms(1,2,:) .* Ms(2,1,:);
    
    siz = size(Ms);
    
    if ndims(Ms) == 3
        siz_r = [siz(3), 1];
    else
        siz_r = siz(3:end);
    end
    
    r = reshape(r, siz_r);
    
end

