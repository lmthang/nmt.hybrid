function D = slhistmetric_pw(H1, H2, mtype, varargin)
%SLHISTMETRIC_PW Computes distance metrics between histograms pairwisely
%
% $ Syntax $
%   - D = slhistmetric_pw(H1, H2, mtype, ...)
%
% $ Arguments $
%   - H1, H2:       The matrices of histogram sets (d1 x n1, d2 x n2)
%   - mtype:        The metric type
%   - D:            The resulting pairwise metric matrix (n1 x n2)
%
% $ Description $
%   - D = slhistmetric_cp(H1, H2, mtype, ...) computes the distance metrics
%     between all histograms in H1 and H2 pairwisely. If H1 and H2 have
%     n1 and n2 bins respectively, then D will be an n1 x n2 matrix.
%     The function support following types of histogram distances:
%     \*
%     \t   Table.  The Histogram Metrics
%     \h      name     &          description
%          'L1dist'    &  The sum of absolute differences: 
%                         d = sum |h1(i) - h2(i)|               \\
%          'L2dist'    &  Euclidean distance: 
%                         d = sqrt(sum( (h1(i) - h2(i))^2 ))    \\
%          'quaddist'  &  Quadratic-Form distance: 
%                         d = sqrt((h1 - h2)^T * Q * (h1 - h2)) \\
%                         use Q (d x d matrix) as the first extra param.
%          'hamming'   &  Hamming distance with threshold
%                         ht1 = h1 > t
%                         ht2 = h2 > t
%                         d = sum(ht1 ~= ht2)                  
%                         use threshold t as the first extra param.\\
%          'intersect' &  Histogram Intersection: 
%                         d = 1 - 
%                         sum min(h1(i), h2(i))) / sum h2(i) \\
%          'chisq'     &  Chi-Square Distance:
%                         d = sum (h1(i) - h2(i))^2 / (2 * (h1(i)+h2(i)) \\
%          'kolsm'     &  Kolmogorov-Smirnov distance: 
%                         d = max |F1(i) - F2(i)| 
%                         only suitable for scalar histogram.   \\
%          'kramvon'   &  Kramer/Von Mises:
%                         d = sum (F1(i) - F2(i))^2              \\
%          'kldiv'     &  Kull-back Leibler divergence
%                         d = sum h1(i) log h1(i) / h2(i)        \\
%          'jeffrey'   &  Jeffrey divergence
%                         d = KL(h1, (h1+h2)/2) + KL(h2, (h1+h2)/2) \\
%     \*     
%          
% $ History $
%   - Created by Dahua Lin, on Sep 18, 2006
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('slhistmetric_cp', 3);
end
if ~isnumeric(H1) || ~isnumeric(H2) || ndims(H1) ~= 2 || ndims(H2) ~= 2
    error('sltoolbox:invalidarg', ...
        'H1 and H2 should be 2D numeric matrices');
end


%% main delegation

switch mtype
    case 'L1dist'
        checkhistdim(H1, H2);
        D = slmetric_pw(H1, H2, 'cityblk');
        
    case 'L2dist'
        checkhistdim(H1, H2);
        D = slmetric_pw(H1, H2, 'eucdist');
                
    case 'quaddist'
        if length(varargin) ~= 1
            error('sltoolbox:invalidarg', ...
                'quaddist has one extra parameter Q');
        end
        checkhistdim(H1, H2);
        D = slmetric_pw(H1, H2, 'quaddiff', varargin{1});
        D = sqrt(D);
        
    case 'hamming'
        checkhistdim(H1, H2);
        if length(varargin) ~= 1
            error('sltoolbox:invalidarg', ...
                'quaddist has one extra parameter t');
        end
        t = varargin{1};
        D = sldiff_pw(double(H1 > t), double(H2 > t), 'abssum');
        
    case 'intersect'
        checkhistdim(H1, H2);
        D = histmetricpw_core(H1, H2, 1);
                
    case 'chisq'
        checkhistdim(H1, H2);
        D = histmetricpw_core(H1, H2, 2);        
        
    case 'kolsm'
        checkhistdim(H1, H2);
        F1 = cumsum(H1, 1);
        F2 = cumsum(H2, 1);
        D = sldiff_pw(F1, F2, 'maxdiff');
        
    case 'kramvon'
        checkhistdim(H1, H2);
        F1 = cumsum(H1, 1);
        F2 = cumsum(H2, 1);
        D = slmetric_pw(F1, F2, 'sqdist');        
        
    case 'kldiv'
        checkhistdim(H1, H2);
        D = kldivergence(H1, H2);
        
    case 'jeffrey'
        checkhistdim(H1, H2);
        D = histmetricpw_core(H1, H2, 3);
                      
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid histogram metric type: %s', mtype);
end
        

%% Auxiliary functions

function d = checkhistdim(H1, H2)

d = size(H1, 1);
if size(H2, 1) ~= d
    error('sltoolbox:sizmismatch', 'H1 and H2 have different dimensions.');
end

function D = kldivergence(H1, H2)

V = zeros(size(H1));
not_zero = H1 > 0;
V(not_zero) = H1(not_zero) .* log(H1(not_zero));
v1 = sum(V, 1)';
clear V;

H2(~not_zero) = 1;
L2 = log(H2);
D = -H1' * L2;
D = sladdvec(D, v1, 1);




