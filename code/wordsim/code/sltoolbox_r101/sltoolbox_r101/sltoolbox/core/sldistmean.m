function m = sldistmean(X1, X2, varargin)
%SLDISTMEAN Uses fast method to compute means of pairwise distances
%
% $ Syntax $
%   - m = sldistmean(X1, X2, mtype, ...)
%   - m = sldistmean(X1, X2, w1, w2, mtype, ...)
% 
% $ Arguments $
%   - X1, X2:       The samples to compute the mean of pairwise distances
%   - mtype:        The distance metric type
%   - w1, w2:       The weights of samples in X1 and X2
%   - m:            The mean value of all pairwise distances
%
% $ Description $
%   - m = sldistmean(X1, X2, mtype, ...) computes the average value of 
%     a specified type of distances by using a mathematical equivalent
%     and fast way. It would be much faster than computing all pairwise
%     distances and calculating their average.
%     \*
%     \t      Table. Supported Distance Types
%     \h       name      &             description 
%             'sqdist'   &  square distances d = ||x1 - x2||^2
%             'wsqdist'  &  weighted square distances:
%                           d = sum_k w_k (x1(k) - x2(k))^2 
%                           the first extra parameter it a column vector
%                           of weights on all components
%             'quaddiff' &  distance in quadratic form:
%                           d = (x1 - x2)^T Q (x1 - x2)
%                           the first extra parameter is a quadratic form
%                           matrix
%     \*
%     Please not that, if X1 and X2 have n1 an n2 samples respectively, then
%     m = 1 /(n1 * n2) * sum_i sum_j d(X1(:,i), X2(:,j)).
%
%   - m = sldistmean(X1, X2, w1, w2, mtype, ...) computes the weighted
%     average value of the distances. The w1 and w2 assign weights to
%     samples in X1 and X2 respectively. They should be row vectors of
%     size 1 x n1 and 1 x n2.
%
% $ History $
%   - Created by Dahua Lin, on Sep 20, 2006
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('sldistmean', 3);
end

if ~isnumeric(X1) || ~isnumeric(X2) || ndims(X1) ~= 2 || ndims(X2) ~= 2
    error('sltoolbox:invalidarg', ...
        'X1 and X2 should be numeric 2D matrices');
end

if ischar(varargin{1})
    mtype = varargin{1};
    if nargin == 3
        params = {};
    else
        params = varargin(2:end);
    end
    w1 = [];
    w2 = [];
else
    w1 = varargin{1};
    w2 = varargin{2};
    
    n1 = size(X1, 2);
    n2 = size(X2, 2);
    if ~isequal(size(w1), [1 n1]) || ~isequal(size(w2), [1, n2])
        error('sltoolbox:invalidarg', ...
            'The size of w1 or w2 is illegal');
    end
    
    mtype = varargin{3};
    if nargin == 5
        params = {};
    else
        params = varargin(4:end);
    end
end


%% main skeleton

switch mtype
    case {'sqdist', 'wsqdist'}
        d = check_samedim(X1, X2);
        if strcmp(mtype, 'sqdist')
            wc = [];
        else
            check_paramsnum(mtype, params, 1);
            wc = params{1};
            if ~isequal(size(wc), [d, 1])
                error('sltoolbox:sizmismatch', ...
                    'The weights on components should be d x 1 vector');
            end
        end                
        
        vm1 = slmean(X1, w1, true);
        vm2 = slmean(X2, w2, true);
        vs1 = compute_vars(X1, vm1, w1);
        vs2 = compute_vars(X2, vm2, w2);
        vmd = vm1 - vm2;
        vsm = vmd .* vmd;
        
        if isempty(wc)
            m = sum(vsm + vs1 + vs2);
        else
            m = sum((vsm + vs1 + vs2) .* wc);
        end
                       
    case 'quaddiff'
        d = check_samedim(X1, X2);
        check_paramsnum(mtype, params, 1);
        Q = params{1};
        if ~isequal(size(Q), [d d])
            error('sltoolbox:sizmismatch', ...
                'The Q matrix should be a d x d square matrix');
        end
        
        vm1 = slmean(X1, w1, true);
        vm2 = slmean(X2, w2, true);
        C1 = slcov(X1, w1, vm1, true);
        C2 = slcov(X2, w2, vm2, true);
        vmd = vm1 - vm2;
        
        m = vmd' * Q * vmd + sum(sum(Q .* (C1 + C2)));
                
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid metric type: %s', mtype);
end

        
%% Auxiliary functions

function d = check_samedim(X1, X2)

d = size(X1, 1);
if d ~= size(X2, 1)
    error('sltoolbox:sizmismatch', ...
        'X1 and X2 have different sample dimensions');
end

function check_paramsnum(name, params, n)

if length(params) ~= n
    error('sltoolbox:invalidarg', ...
        'For metric %s, it has %d extra parameters.', name, n);
end

function vs = compute_vars(X, vmean, w)

DX = sladdvec(X, -vmean);
vs = slmean(DX .* DX, w);








