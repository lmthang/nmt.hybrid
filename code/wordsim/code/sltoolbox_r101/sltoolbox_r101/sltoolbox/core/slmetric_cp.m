function M = slmetric_cp(X1, X2, mtype, varargin)
%SLMETRIC_CP Computes the metrics between corresponding pairs of samples
%
% $ Syntax $
%   - M = slmetric_cp(X1, X2, mtype, ...);
%
% $ Arguments $
%   - X1, X2:       The sample matrices with each column being a sample
%   - mtype:        The metric type
%   - M:            The computed results
%
% $ Arguments $
%   - M = slmetric_cp(X1, X2, mtype, ...) computes the metrics between
%     corresponding pairs of samples given in X1 and X2. X1 and X2 should
%     have the same number of columns, say n. Then M would be a 1 x n
%     row vector. 
%
%    - The supported metrics of this function are listed as follows:
%      \*
%      \t  Table 1. The supported metrics                             \\
%      \h     name     &       description                            \\
%          'eucdist'   &  Euclidean distance: ||x - y||               \\         
%          'sqdist'    &  Square of Euclidean distance: ||x - y||^2   \\
%          'dotprod'   &  Canonical dot product: <x,y> = x^T * y      \\
%          'nrmcorr'   &  Normalized correlation (cosine angle):
%                         (x^T * y ) / (||x|| * ||y||)                \\
%          'angle'     &  Angle between two vectors (in radian)       \\
%          'quadfrm'   &  Quadratic form:  x^T * Q * y                
%                         Q is specified in the 1st extra parameter   \\
%          'quaddiff'  &  Quadratic form of difference:
%                         (x - y)^T * Q * (x - y),                
%                         Q is specified in the 1st extra parameter   \\
%          'cityblk'   &  City block distance (abssum of difference)  \\
%          'maxdiff'   &  Maximum absolute difference                 \\
%          'mindiff'   &  Minimum absolute difference                 \\
%          'wsqdist'   &  Weighted square of Euclidean distance       \\
%                         \sum_i w_i (x_i - y_i)^2,  w = (w_1, ..., w_d)                     
%                         the weights w is specified in 1st extra parameter 
%                         as a length-d column vector                  \\
%      \*
%
% $ History $
%   - Created by Dahua Lin, on Sep 4th, 2006
%

%% Parse and verify input arguments

if nargin < 3
    raise_lackinput('slmetric_cp', 3);
end

if ~isnumeric(X1) || ~isnumeric(X2) || ndims(X1) ~= 2 || ndims(X2) ~= 2
    error('sltoolbox:invalidarg', ...
        'The X1 and X2 should be numeric 2D matrix');
end

n = size(X1, 2);
if size(X2, 2) ~= n
    error('sltoolbox:sizmismatch', ...
        'X1 and X2 have different numbers of samples');
end

%% Main skeleton

switch mtype
    case {'eucdist', 'sqdist'}
        check_samedim(X1, X2);
        D = X1 - X2;
        M = sum(D .* D, 1);
        M(M < 0) = 0;
        if strcmp(mtype, 'eucdist')
            M = sqrt(M);
        end
        
    case 'dotprod'
        check_samedim(X1, X2);
        M = sum(X1 .* X2, 1);
        
    case {'nrmcorr', 'angle'}
        check_samedim(X1, X2);
        M = sum(X1 .* X2, 1);
        N1 = sum(X1 .* X1, 1);
        N2 = sum(X2 .* X2, 1);
        N1(N1 < 0) = 0;
        N2(N2 < 0) = 0;
        M = M ./ (sqrt(N1) .* sqrt(N2));
        if strcmp(mtype, 'angle')
            M = real(acos(M));
        end
        
    case 'quadfrm'
        d1 = size(X1, 1);
        d2 = size(X2, 1);
        Q = varargin{1};
        if ~isequal(size(Q), [d1, d2])
            error('sltoolbox:sizmismatch', ...
                'The size of Q is not consistent with the samples');
        end
        QX2 = Q * X2;
        M = sum(X1 .* QX2, 1);
        
    case 'quaddiff'
        d = check_samedim(X1, X2);
        Q = varargin{1};
        if ~isequal(size(Q), [d, d])
            error('sltoolbox:sizmismatch', ...
                'The size of Q is not consistent with the samples');
        end
        D = X1 - X2;
        QD = Q * D;
        M = sum(D .* QD, 1);
        
    case 'cityblk'
        check_samedim(X1, X2);
        D = X1 - X2;
        M = sum(abs(D), 1);
        
    case 'maxdiff'
        check_samedim(X1, X2);
        D = X1 - X2;
        M = max(abs(D), [], 1);
        
    case 'mindiff'
        check_samedim(X1, X2);
        D = X1 - X2;
        M = min(abs(D), [], 1);
            
    case 'wsqdist'
        d = check_samedim(X1, X2);
        w = varargin{1};
        if ~isequal(size(w), [d, 1])
            error('sltoolbox:sizmismatch', ...
                'w should be a d x 1 vector');
        end
        D = X1 - X2;
        D2 = D .* D;
        clear D;
        D2 = slmulvec(D2, w, 1);
        M = sum(D2, 1);      
        
    otherwise
        error('sltoolbox:invalidarg', ...
            'Unknown metric type: %s', mtype);
        
end


%% Auxiliary functions

function d = check_samedim(X1, X2)

d = size(X1, 1);
if d ~= size(X2, 1)
    error('sltoolbox:sizmismatch', ...
        'X1 and X2 have different dimensions');
end











