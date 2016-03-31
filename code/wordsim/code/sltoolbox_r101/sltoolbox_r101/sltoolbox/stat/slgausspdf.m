function P = slgausspdf(GS, X, varargin)
%SLGAUSSPDF Computes the probability density of Gaussian models
%
% $ Syntax $
%   - P = slgausspdf(GS, X, ...)
%
% $ Arguments $
%   - GS:       The Gaussian model struct
%   - X:        the sample matrix
%   - P:        the computed results
%
% $ Description $
%   - P = slgausspdf(GS, X, ...) computes the pdf of Gaussian distribution
%     on the samples given in X. If there are k models in GS and n samples
%     in X, then P would be a k x n matrix, with each column corresponding
%     to a sample. You can also specify following properties 
%       - 'output':     the type of output values in P
%                       - 'normal': the pdf (default)
%                       - 'log': the logarithm of the pdf
%                       - 'neglog': the negation of logarithm of the pdf
%
% $ Remarks $
%   - The implementation of this function is based on slgaussmdist
%
% $ History $
%   - Created by Dahua Lin, on Aug 28, 2006
%   - Modified by Dahua Lin, on Sep 10, 2006
%       - replace sladd by sladdvec to increase efficiency
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slgausspdf', 2);
end

tyi = slgausstype(GS);

opts.output = 'normal';
opts = slparseprops(opts, varargin{:});

if ~ismember(opts.output, {'normal', 'log', 'neglog'})
    error('sltoolbox:invalidarg', ...
        'Invalid output form %s', opts.output);
end

d = GS.dim;

%% compute

% compute Mahalanobis distance

mdists = slgaussmdist(GS, X);
md2 = mdists .* mdists;
clear mdists;

% compute other terms

switch tyi.varform
    case 'univar'
        vt = varterm_univar(d, GS.vars);
    case 'diagvar'
        vt = varterm_diagvar(GS.vars);
    case 'covar'
        vt = varterm_covar(GS.covs);
end

if tyi.sharevar        
    a = d * log(2*pi) + vt;
    P = md2 + a;
    clear md2;        
else % not sharevar        
    a = d * log(2*pi) + vt;
    P = sladdvec(md2, a, 1);
    clear md2;                       
end

P = 0.5 * P;


%% convert output

switch opts.output
    case 'log'
        P = -P;
    case 'normal'
        P = exp(-P);        
end




%% The functions to compute model variance terms

function t = varterm_univar(d, v)

t = d * log(v)';

function t = varterm_diagvar(v)

t = sum(log(v), 1)';

function t = varterm_covar(C)

k = size(C, 3);
if k == 1
    t = sllogdet(C);
else
    t = zeros(k, 1);
    for i = 1 : k
        t(i) = sllogdet(C(:,:,i));
    end
end
    




