function R = slgaussinv(GS, fn, invparams)
%SLGAUSSINV Computes the inverse of variance/covariance in Gaussian model
%
% $ Syntax $
%   - R = slgaussinv(GS, fn, invparams)
%
% $ Arguments $
%   - GS:           The Gaussian struct
%   - fn:           the fieldname representing thr variance: 'vars'|'covs'
%   - invparams:    the cell array of parameters for computing inverses
%   - R:            the computed result
%
% $ Description $
%   - R = slgaussinv(GS, fn, invparams) computes the inverse of variances or 
%     covariances for Gaussian models. 
%
% $ Remarks $
%   - For vars, the inverses will be computed using slinvevals, while
%     for covs, the inverses will be computed using slinvcov.
%
% $ History $
%   - Created by Dahua Lin, on Aug 26, 2006
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('slgaussinv', 3);
end

if ~isstruct(GS)
    error('sltoolbox:invalidarg', ...
        'GS should be a struct');
end

if ~isfield(GS, fn)
    error('sltoolbox:invalidarg', ...
        '%s should be a field of GS', fn);
end

%% computation

switch fn
    case 'vars'
        vars = GS.vars;
        [dv, kv] = size(vars);
        if dv == 1
            if kv == 1
                invvars = 1 / vars;
            else
                invvars = 1 ./ vars;
            end
        else
            if kv == 1
                invvars = slinvevals(vars, invparams{:});
            else
                invvars = zeros(dv, kv);
                for i = 1 : kv
                    invvars(:,i) = slinvevals(vars(:,i), invparams{:});
                end
            end
        end
        R = invvars;
        
    case 'covs'
        covs = GS.covs;
        invcovs = slinvcov(covs, invparams{:});
        R = invcovs;
        
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid variance/covariance fieldname %s', fn);
end
        
        
        
        
