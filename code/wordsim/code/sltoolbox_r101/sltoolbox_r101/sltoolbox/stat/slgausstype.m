function tyinfo = slgausstype(GS)
%SLGAUSSTYPE Judges the type of a Gaussian model struct
%
% $ Syntax $
%   - tyinfo = slgausstype(GS)
%
% $ Arguments $
%   - GS:       the Gaussian model struct
%   - tyinfo:   the type information structure with following fields
%               - varform:  the form of variance: 'univar'|'diagvar'|'covar'
%               - sharevar: whether the variance(covariance) is shared
%               - hasinv:   whether is invvars or invcovs exists
%               - hasmw:    whether the mixture weights exist
%
% $ Remarks $
%   - The function will check the validity of the struct and will raise
%     an error for invalid models. So it can be used to check validity
%     even you don't need to know the type of the model.
%
% $ History $
%   - Created by Dahua Lin, on Aug 23rd, 2006
%


%% verify basic fields

if ~isstruct(GS)
    gs_argerror('The Gaussian model should be a struct');
end

if ~all(slisfields(GS, {'dim', 'nmodels', 'means'}))
    gs_argerror('The Gaussian model struct should have all of the fields: dim, nmodels and means');
end

d = GS.dim;
k = GS.nmodels;

if ~isequal(size(GS.means), [d k])
    gs_sizerror('The means field should be an array of size d x k');
end

%% verify variance/covariance field

if isfield(GS, 'vars')
    sizvf = size(GS.vars);
    if isequal(sizvf, [1 1])
        varform = 'univar';
        sharevar = true;
    elseif isequal(sizvf, [1 k])
        varform = 'univar';
        sharevar = false;
    elseif isequal(sizvf, [d 1])
        varform = 'diagvar';
        sharevar = true;
    elseif isequal(sizvf, [d k])
        varform = 'diagvar';
        sharevar = false;
    else
        gs_arrerror('The size of vars is illegal');
    end
    
    hasinv = isfield(GS, 'invvars');    
    
elseif isfield(GS, 'covs')
    
    sizcvf = size(GS.covs);
    if isequal(sizcvf, [d d])
        varform = 'covar';
        sharevar = true;
    elseif isequal(sizcvf, [d d k])
        varform = 'covar';
        sharevar = false;
    else
        gs_arrerror('The size of covariance is illegal');
    end
    
    hasinv = isfield(GS, 'invcovs'); 
    
else
    gs_arrerror('The Gaussian struct lacks a field for variance/covariance');
end


hasmw = isfield(GS, 'mixweights');

if nargout >= 1 
    tyinfo = struct(...
        'varform', varform, ...
        'sharevar', sharevar, ...
        'hasinv', hasinv, ...
        'hasmw', hasmw);
end



function gs_argerror(errmsg)

error('sltoolbox:invalidarg', errmsg);

function gs_sizerror(errmsg)

error('sltoolbox:sizmismatch', errmsg);
