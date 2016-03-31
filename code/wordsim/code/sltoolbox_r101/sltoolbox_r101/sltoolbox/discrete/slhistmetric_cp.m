function D = slhistmetric_cp(H1, H2, mtype, varargin)
%SLHISTMETRIC_CP Computes the metrics between corresponding pairs of histograms  
%
% $ Syntax $
%   - D = slhistmetric_cp(H1, H2, mtype, ...)
%
% $ Arguments $
%   - H1, H2:       The histograms for metric computing (d x n)
%   - mtype:        The metric type
%   - D:            The resultant vector (1 x n)
%
% $ Description $
%   - D = slhistmetric_cp(H1, H2, mtype, ...) computes the distance metrics
%     between corresponding pairs of histograms in H1 and H2.
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
n = size(H1, 2);
if size(H2, 2) ~= n
    error('sltoolbox:sizmismatch', ...
        'H1 and H2 have different numbers of histograms');
end


%% main delegation

switch mtype
    case 'L1dist'
        checkhistdim(H1, H2);
        D = slmetric_cp(H1, H2, 'cityblk');
        
    case 'L2dist'
        checkhistdim(H1, H2);
        D = slmetric_cp(H1, H2, 'eucdist');
                
    case 'quaddist'
        if length(varargin) ~= 1
            error('sltoolbox:invalidarg', ...
                'quaddist has one extra parameter Q');
        end
        checkhistdim(H1, H2);
        D = slmetric_cp(H1, H2, 'quaddiff', varargin{1});
        D = sqrt(D);
        
    case 'hamming'
        checkhistdim(H1, H2);
        if length(varargin) ~= 1
            error('sltoolbox:invalidarg', ...
                'quaddist has one extra parameter t');
        end
        t = varargin{1};
        D = sum((H1 > t) ~= (H2 > t), 1);
        
    case 'intersect'
        checkhistdim(H1, H2);
        D = 1 - sum(min(H1, H2), 1) ./ sum(H2, 1);
                
    case 'chisq'
        checkhistdim(H1, H2);
        D = H1 - H2;
        S = H1 + H2;
        D = sum(D.*D ./ (2 * S), 1);
        
    case 'kolsm'
        checkhistdim(H1, H2);
        F1 = cumsum(H1, 1);
        F2 = cumsum(H2, 1);
        D = max(abs(F1 - F2), [], 1);
        
    case 'kramvon'
        checkhistdim(H1, H2);
        F1 = cumsum(H1, 1);
        F2 = cumsum(H2, 1);
        D = F1 - F2;
        D = sum(D .* D, 1);
        
    case 'kldiv'
        checkhistdim(H1, H2);
        D = kldivergence(H1, H2);
        
    case 'jeffrey'
        checkhistdim(H1, H2);
        Ha = (H1 + H2) / 2;
        V1 = sum(mullog(H1, H1), 1);
        V2 = sum(mullog(H2, H2), 1);
        V12 = sum(mullog(Ha, Ha), 1);
        D = V1 + V2 - 2 * V12;
                
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

function d = kldivergence(H1, H2)

d = sum(mullog(H1, H1), 1) - sum(mullog(H1, H2), 1);


function V = mullog(H1, H2)
% compute H1 .* log(H2) in a robust way

V = zeros(size(H1));
not_zero = H1 > 0;
V(not_zero) = H1(not_zero) .* log(H2(not_zero));


