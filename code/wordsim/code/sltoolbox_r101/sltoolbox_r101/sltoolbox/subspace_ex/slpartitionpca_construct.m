function A = slpartitionpca_construct(S, modeldir, feas)
%SLPARTITIONPCA_CONSTRUCT Constructs the array from features 
%
% $ Syntax $
%   - A = slpartitionpca_construct(S, modeldir, feas)
%
% $ Arguments $
%   - S:            the partition-based PCA model struct or model core file
%   - modeldir:     the directory path of the projection files
%   - feas:         the partition PCA features
%   - A:            the constructed arrays in original space
%
% $ Description $
%   - A = slpartitionpca_construct(S, modeldir, feas) constructs the 
%     array units in the original space from the extracted partiton PCA
%     features. Here, S can be a struct loaded from core file or the core 
%     filename. modeldir is the directory of the corefile and the related 
%     projection matrix files.
%
% $ Remarks $
%   - When the combined model is learned, the dimension of features can be
%     lower than the dimension of feature subspace of the PCA model. In 
%     this case, only the first k principal components are used.
%
% $ History $
%   - Created by Dahua Lin, on Jul 30th, 2006
%   - Modified by Dahua Lin, on Sep 10, 2006
%       - replace sladd by sladdvec to increase efficiency
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('slpartitionpca_apply', 3);
end

if ischar(S)
    S = load(S);
elseif ~isstruct(S)
    error('sltoolbox:invalidarg', ...
        'The S should be the filename of the core file or the core struct');
end

[k, n] = size(feas);
if k > S.diminfo.feadim
    error('sltoolbox:sizmismatch', ...
        'k is larger than the dimension of feature space');
elseif k < S.diminfo.feadim && isempty(S.combprojfile)
    error('sltoolbox:sizmismatch', ...
        'When the combined model is not learned, k should be exactly the same as the dimension of feature space');
end

%% Construct Intermediate features

if ~isempty(S.combprojfile)
   
    combprojpath = sladdpath(S.combprojfile, modeldir);
    P = slreadarray(combprojpath);
    if k < S.diminfo.feadim
        P = P(:, 1:k);
    end
    intfeas = P * feas;
    clear P;
    
else
    intfeas = feas;
    
end
    
%% Construct Array Units

A = zeros([size(S.meanarr), n]);
NBlks = numel(S.blocks);
projpaths = sladdpath(S.projfiles, modeldir);

dc = 0;
for ib = 1 : NBlks
    
    curblock = S.blocks{ib};
    rgncell = slrange2indcells(curblock);
    
    cursiz = curblock(2,:) - curblock(1,:) + 1;
    curdim = prod(cursiz);
    cursubdim = S.diminfo.subdims(ib);
    
    curfeasec = intfeas(dc+1:dc+cursubdim, :);        
    curproj = slreadarray(projpaths{ib});
    
    localmean = S.meanarr(rgncell{:});
    localmean = reshape(localmean, [curdim, 1]);
    
    localarr = curproj * curfeasec;
    clear curproj;
    
    localarr = sladdvec(localarr, localmean, 1);
    localarr = reshape(localarr, [cursiz, n]);
    
    A(rgncell{:}, :) = localarr;    
    
    clear localarr localmean;
    
    dc = dc + cursubdim;
    
end




    
    
