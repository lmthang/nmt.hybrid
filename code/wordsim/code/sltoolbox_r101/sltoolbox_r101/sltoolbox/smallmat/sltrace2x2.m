function r = sltrace2x2(Ms)
%SLTRACE2X2 Computes the trace of 2 x 2 matrices in a fast way
%
% $ Syntax $
%   - r = sltrace2x2(Ms)
%
% $ Arguments $
%   - Ms:       the 2 x 2 matrix (matrices)
%   - r:        the resultant trace(s)
%
% $ Description $
%   - r = sltrace2x2(Ms) computes the trace of 2x2 matrices. If Ms is
%     a 2 x 2 matrix, then r is a scalar representing its trace. 
%     Or, if Ms is a 2 x 2 x ... array, then r would be a ... array 
%     storing the trace of all 2 x 2 matrices.
%
% $ Remarks $
%   - The function uses the following formula for fast calculation of
%     trace of 2 x 2 matrices:
%       det = a11 + a22
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
    r = Ms(1) + Ms(4);
    
else                    % a set of matrices
    r = Ms(1,1,:) + Ms(2,2,:);
    
    siz = size(Ms);
    
    if ndims(Ms) == 3
        siz_r = [siz(3), 1];
    else
        siz_r = siz(3:end);
    end
    
    r = reshape(r, siz_r);
    
end

