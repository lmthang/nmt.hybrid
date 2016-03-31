function feas = slpartitionpca_apply(S, modeldir, data, n, k)
%SLPARTITIONPCA_APPLY applies partition-based PCA to a set of arrays
%
% $ Syntax $
%   - feas = slpartitionpca_apply(S, modeldir, data, n)
%   - feas = slpartitionpca_apply(S, modeldir, data, n, k)
%
% $ Arguments $
%   - S:            the partition-based PCA model struct or model core file
%   - modeldir:     the directory path of the projection files
%   - data:         the arrays for feature extraction
%   - n:            the number of samples
%   - k:            the finally used number of features
%   - feas:         the features
%
% $ Description $
%   - feas = slpartitionpca_apply(S, modeldir, data, n) applies the 
%     partition PCA to a set of array units given in data. Here, S can be 
%     a struct loaded from core file or the core filename. modeldir is the 
%     directory of the corefile and the related projection matrix files.
%     data can be either an array or a cell array of array filenames.
%
%   - feas = slpartitionpca_apply(S, modeldir, data, n, k) further 
%     reduces the feature space dimension to k by discarding the trailing
%     feature components. The feature space can only be truncated when
%     the combined model is learned.
%
% $ History $
%   - Created by Dahua Lin on Jul 30th, 2006
%   - Modified by Dahua Lin, on Sep 10, 2006
%       - replace sladd by sladdvec to increase efficiency
%

%% parse and verify input arguments

if nargin < 4
    raise_lackinput('slpartitionpca_apply', 4);
end

if ischar(S)
    S = load(S);
elseif ~isstruct(S)
    error('sltoolbox:invalidarg', ...
        'The S should be the filename of the core file or the core struct');
end

if ~isnumeric(data) && ~iscell(data)
    error('sltoolbox:invalidarg', ...
        'data should be an numeric array or a cell array of strings');
end

if nargin < 5
    k = S.diminfo.feadim;
else
    if k > S.diminfo.feadim
        error('k is higher than the dimension of the whole feature subspace');
    elseif k < S.diminfo.feadim && isempty(S.combprojfile)
        error('The feature space cannot be truncated without the combined model');
    end
end


%% Generate the intermediate stacked features

NBlks = numel(S.blocks);
intfeas = zeros(S.diminfo.intdim, n);

projpaths = sladdpath(S.projfiles, modeldir);

dc = 0;
for ib = 1 : NBlks
    
    curblock = S.blocks{ib};
    cursubdim = S.diminfo.subdims(ib);    
    rgncell = slrange2indcells(curblock);
    
    projmat = slreadarray(projpaths{ib});
    
    V = generate_intfeature(S, data, n, rgncell, projmat);
    intfeas(dc+1:dc+cursubdim, :) = V;
    dc = dc + cursubdim;
    
    clear projmat;    
end


%% Apply combined projection

if ~isempty(S.combprojfile)
    combprojpath = sladdpath(S.combprojfile, modeldir);
    
    P = slreadarray(combprojpath);    
    if k < S.diminfo.feadim
        P = P(:, 1:k);
    end    
    P = P';
    
    feas = P * intfeas;
else
    feas = intfeas;
end
            


%% Core computing functions

function V = generate_intfeature(S, data, n, rangecell, projmat)

[d0, d1] = size(projmat);
nd = length(rangecell);

if isnumeric(data)
    
    if size(data, nd+1) ~= n
        error('sltoolbox:sizmismatch', ...
            'The data size does not match the specified sample number');
    end
    
    localarr = data(rangecell{:}, :);
    localarr = reshape(localarr, [d0, n]);
    localmean = S.meanarr(rangecell{:}, :);
    localmean = reshape(localmean, [d0, 1]);
    
    localarr = sladdvec(localarr, -localmean, 1);
    
    V = projmat' * localarr;
    
else
    
    V = zeros(d1, n);
    
    cf = 0;
    nfiles = length(data);
    for i = 1 : nfiles        
        datapart = slreadarray(data{i});
        curn = size(datapart, nd+1);        
        V(:, cf+1:cf+curn) = generate_intfeature(S, datapart, curn, rangecell, projmat);      
        cf = cf + curn;
    end
    
    if cf ~= n
        error('sltoolbox:sizmismatch', ...
            'The total number of units in the set of array files is not n');
    end
end



    
        
   
