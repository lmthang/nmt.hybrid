function slpartitionpca(data, arrsiz, n, ps, filepath, varargin)
%SLPARTITIONPCA Performs Partition-based PCA and saves the models
%
% $ Syntax $
%   - slpartitionpca(data, arrsiz, n, ps, filepath, ...)
%   
% $ Arguments $
%   - data:         the super-array of the unit arrays, or the set of
%                   filenames storing the arrays.
%   - arrsiz:       the size of each unit array
%   - n:            the number of samples
%   - ps:           the partition structure for each unit
%   - filepath:     the destination filepath (without extension)
%   
% $ Description $
%   - slpartitionpca(data, arrsiz, n, ps, filepath, ...) applies PCA to 
%     large arrays. To make the computation tractable, it divides the 
%     whole matrix into several partitions according to the structure
%     specified in ps. The trained models be stored in filepath and
%     related files. 
%     It will creates a core file named filepath.mat to give basic
%     information of the PCA model, which contains the following variables:
%       - 'parstruct':      the parition structure
%       - 'blocks':         the cell array of specification of blocks
%                           if the partition structure divides the whole
%                           array unit into m1 x m2 x ... blocks, then
%                           blocks would be a m1 x m2 x ... cell array,
%                           with each cell being a 2 x d matrix in the form
%                           of [s1 s2, ... sd; e1, e2, ..., ed]
%                           then the actual block extracted from an array
%                           unit A would be A(s1:e1, s2:e2, ..., sd:ed)
%       - 'projfiles'       The cell array of projection array files 
%                           corresponding to the blocks.
%       - 'combprojfile'    The filename of the combined projection. If
%                           the combined model is not learned, this
%                           filename is empty.
%       - 'meanarr'         The mean array
%       - 'energy'          The structure representing the energy info
%                           - 'total': the original total energy;
%                           - 'intpreserved': the total preserved energy of
%                             all partitions
%                           - 'combpreserved': the preserved energy of
%                             combination model
%                           - 'par':  the original partition enegies:
%                              an m1 x m2 x ... array.
%                           - 'parpreserved': the preserved partition 
%                              energy. an m1 x m2 x ... array.
%       - 'diminfo'         The information of the space dimension
%                           - 'size':   the size of the whole array unit
%                           - 'oridim': the original total dimension
%                           - 'dims':   the original dimensions of the 
%                             partitions: an m1 x m2 x ... array.
%                           - 'subdims': the subspace dimension of the
%                             partitions: an m1 x m2 x ... array
%                           - 'intdim': the dimension of the intermediate
%                             stacked vector space.
%                           - 'feadim': the dimension of the combined
%                             subspace. (If the combined model is not
%                             learned, subdim = intdim)
%       - 'evalset'         The eigenvalues preserved for all partitions
%       - 'combevals'       The eigenvalues of the combination model
%
%     The projection matrices will be stored in a set of array files 
%     named as filepath.proj.01, ...., the the combined model is learned
%     its projection matrix is given by filepath.proj.comb.
%     In addition, you can specify following properties to control the
%     learning of partition-based PCA.
%     \*
%     \t    Properties of Partition-based PCA Learning
%     \h     name        &      description
%           'combmodel'  &  Whether to learn a combined PCA model, 
%                           default = true;
%           'er0'        &  The level-0 of energy preservation ratio
%                           (default = 0.99)
%           'er1'        &  The level-1 of energy preservation ratio 
%                           (default = 0.95, only takes effect in the 
%                            training of combined model)
%           'mixev'      &  Whether to mix up the eigenvalues of all PCA
%                           models when selecting the principal components
%                           (default = 0)
%           'meanarr'    &  The precomputed mean array (default = [])
%           'weights'    &  The weights of individual samples 
%                           (default = [])
%           'verbose'    &  Whether to show intermediate step information
%                           (default = true)
%     \*
%   
%
% $ Remarks $
%   - The algorithm implements a divide and conquer strategy, it first
%     divides the whole array unit into smaller partitions, then train PCA
%     for each partition. The principal components for all the partitions 
%     are stacked together and then a combined PCA model is trained based 
%     on the stacked space. The combined model is learned when 
%   - On the selection of principal components for individual partitions,
%     there are two strategies. The simpler one is the separate strategy,
%     that is to select principal components merely based on the
%     eigenspectrum of the parition PCA model itself and select the 
%     components corresponding to the largest eigenvalues so that the 
%     energy of this partition is preserved up to the ratio of er0.
%     Another strategy is mix-up strategy, which pools all eigenvalues
%     from all paritions together, and select the eigenvectors 
%     corresponding to the largest eigenvalues according to the overall
%     ranking. It can be proved that the mix-up strategy is more efficient.
%     The property mixev is to balance the two strategies, its value
%     ranges in [0, 1]. When mixev is 0, then purely separate strategy 
%     would be used; when mixev is 1, then purely mix-up strategy would
%     be used. When 0 < mixev < 1, we first use separate strategy to 
%     guarantee that each partition preserve up to (1 - mixev) of energy, 
%     then the other mixev of total energy is preserved by pooling all the
%     rest components and select according to overall ranking.
%
% $ History $
%   - Created by Dahua Lin, on Jul 29th, 2006
%   - Modified by Dahua Lin, on Sep 10th, 2006
%       - replace sladd by sladdvec to increase efficiency
%

%% parse and verify input arguments

if nargin < 5
    raise_lackinput('slpartitionpca', 5);
end
arrsiz = arrsiz(:)';
if ~ischar(filepath)
    error('sltoolbox:invalidarg', ...
        'The filepath should be a char string');
end
arrdim = length(arrsiz);

if length(ps) ~= arrdim
    error('sltoolbox:sizmismatch', ...
        'The dimension of partition structure is not consisitent with the array unit');
end

opts.combmodel = true;
opts.er0 = 0.99;
opts.er1 = 0.95;
opts.mixev = 0;
opts.meanarr = [];
opts.weights = [];
opts.verbose = true;
opts = slparseprops(opts, varargin{:});

check_valuerange(opts.er0, 'er0', 0, 1);
check_valuerange(opts.er1, 'er1', 0, opts.er0);
check_valuerange(opts.mixev, 'mixev', 0, 1);

meanarr = opts.meanarr;
if ~isempty(meanarr)
    if ~isequal(size(meanarr), arrmatdim)
        error('sltoolbox:sizmismatch', ...
            'The mean array offered is not consistent with the array unit size');
    end
end

if ~isempty(opts.weights)
    opts.weights = opts.weights(:);
    if length(opts.weights) ~= n
        error('sltoolbox:sizmismatch', ...
            'The length of given weights is inconsistent with the number of samples');
    end
    hasweights = true;
else
    hasweights = false;
end
    


%% Initialization on Partitions and Filenames

showinfo('Initializing Blocks ...', opts);

% partition structure
parstruct = ps;

blocks =  slparblocks(parstruct);
blkinds = slallsubinds(size(blocks));
NBlks = numel(blocks);

projfiles = cell(size(blocks));
for i = 1 : NBlks
    projfiles{i} = [filepath, '.proj.', generate_indexstring(size(blocks), blkinds(:, i)')];
end

if opts.combmodel
    combprojfile = [filepath, '.proj.comb'];
else
    combprojfile = [];
end

clear curinds currange;


%% Prepare Info Structures

showinfo('Preparing Info Structure ...', opts);

energy.total = 0;
energy.intpreserved = 0;
energy.combpreserved = 0;
energy.par = zeros(size(blocks));
energy.parpreserved = zeros(size(blocks));

diminfo.size = arrsiz;
diminfo.oridim = prod(arrsiz);
diminfo.dims = zeros(size(blocks));
diminfo.subdims = zeros(size(blocks));
diminfo.intdim = 0;
diminfo.feadim = 0;



%% Compute the mean array

if isempty(meanarr)
    
    showinfo('Computing Mean Array ...', opts);
    
    if ~hasweights
        meanarr = slarrmean(data, arrsiz, n);
    else
        meanarr = slarrmean(data, arrsiz, n, 'weights', opts.weights);
    end
end


%% Learn the individual PCA models

showinfo('Learning Individual PCA Models ...', opts);

evalset = cell(size(blocks));
for ib = 1 : NBlks
    
    showinfo(sprintf('   PCA Model %d', ib), opts);   
    
    curblk = blocks{ib};
    rgncell = slrange2indcells(curblk);
    vecd = prod(curblk(2,:) - curblk(1,:) + 1);
    
    % compute covariance
    blkcov = compute_localcov(data, meanarr, rgncell, vecd, n, opts.weights);
    
    % compute eigen spectrum
    [evals, evecs] = slsymeig(blkcov);
    
    % initial truncate
    currk = sum(evals >= eps(evals(1)) / 10);    
    evals = evals(1:currk);
    evecs = evecs(:, 1:currk);
                      
    % initial save
    evalset{ib} = evals; 
    slwritearray(evecs, projfiles{ib});
    
    energy.par(ib) = sum(evals);
    diminfo.dims(ib) = vecd; 
    
end

energy.total = sum(energy.par(:));


%% Principal Components Selection

showinfo('Performing Principal Components Selection ...', opts);

if opts.mixev == 0 % separate strategy
    
    showinfo('   Use Separate Strategy', opts);
    
    for ib = 1 : NBlks
                
        evals = evalset{ib};
        currk = decide_rank(evals, opts.er0);     
        
        evals = evals(1:currk);
        evalset{ib} = evals;
        
        evecs = slreadarray(projfiles{ib});
        evecs = evecs(:, 1:currk);
        slwritearray(evecs, projfiles{ib});
                        
        energy.parpreserved(ib) = sum(evals);
        diminfo.subdims(ib) = currk;
        
        clear evals evecs
    end               
    
else % using mix-up strategy
    
    showinfo('   Use Mix-Up Strategy', opts);
    
    showinfo('   Individual Preservation', opts);
    
    % preserve individual projections
    if opts.mixev < 1
        er0 = opts.er0 * (1 - opts.mixev);              
        for ib = 1 : NBlks
            evals = evalset{ib};
            diminfo.subdims(ib) = decide_rank(evals, er0);            
            energy.parpreserved(ib) = sum(evals(1:diminfo.subdims(ib)));
        end        
    else
        diminfo.subdims(:) = 1;
        for ib = 1 : NBlks
            evals = evalset{ib};
            energy.parpreserved(ib) = evals(1);
        end
    end
        
    indv_preserved = sum(energy.parpreserved(:));
    
    % add additional components from mixed pool    
    if indv_preserved < energy.total * opts.er0
        
        showinfo('   Collecting Information for Mix-up ...', opts);
        
        % analyze target
        rest_target_energy = energy.total * opts.er0 - indv_preserved;
        
        % collect rest eigenvalues
        restevnums = zeros(NBlks, 1);
        for ib = 1 : NBlks
            evals = evalset{ib};
            restevnums(ib) = length(evals) - diminfo.subdims(ib);
        end
        total_restevnum = sum(restevnums(:));
        
        evspool = zeros(total_restevnum, 1);
        evsbid = zeros(total_restevnum, 1);
        cn = 0;
        for ib = 1 : NBlks
            if restevnums(ib) > 0
                evals = evalset{ib};
                evspool(cn+1:cn+restevnums(ib)) = ...
                    evals(diminfo.subdims(ib)+1:end);
                evsbid(cn+1:cn+restevnums(ib)) = ib;
                cn = cn + restevnums(ib);
            end            
        end                
        
        showinfo('   Overall Ranking ...', opts);
        
        % overall ranking
        [evspool, sortord] = sort(evspool, 'descend');
        evsbid = evsbid(sortord);
        
        % threshold
        er0 = rest_target_energy / sum(evspool);
        rk = decide_rank(evspool, er0);
        
        showinfo('   Updating Selection ...', opts);
        
        % update selection
        evsbid = evsbid(1:rk);
        for ib = 1 : NBlks
            knew = sum(evsbid == ib);
            if knew > 0
                evals = evalset{ib};
                diminfo.subdims(ib) = diminfo.subdims(ib) + knew;
                energy.parpreserved(ib) = sum(evals(1:diminfo.subdims(ib)));
                evalset{ib} = evals(1:diminfo.subdims(ib));
            end
        end
        
        % truncate the projections
        
        showinfo('   Truncating Projections ...', opts);
        for ib = 1 : NBlks
            pfn = projfiles{ib};
            curprojmat = slreadarray(pfn);
            curprojmat = curprojmat(:, 1:diminfo.subdims(ib));
            slwritearray(curprojmat, pfn);            
        end
        
        clear restevnums total_restevnum rest_target_energy;
        clear evsbid evspool sortord;
                
    end   
            
end

energy.intpreserved = sum(energy.parpreserved(:));
diminfo.intdim = sum(diminfo.subdims(:));


%% Combined PCA model

if opts.combmodel
    
    showinfo('Learning Combined PCA Model ...', opts);
    
    showinfo('   Generating Intermediate Stacked Vectors ...', opts);
    
    % generate intermediate stacked vectors
    intvecs = zeros(diminfo.intdim, n);
    dc = 0;
    for ib = 1 : NBlks
        
        curblk = blocks{ib};
        rgncell = slrange2indcells(curblk);
        vecd = prod(curblk(2,:) - curblk(1,:) + 1);
        cursubdim = diminfo.subdims(ib);    
        
        projmat = slreadarray(projfiles{ib});
        
        V = generate_pvecs(data, rgncell, vecd, cursubdim, n, meanarr, projmat);
        
        intvecs(dc+1:dc+cursubdim, :) = V;
        dc = dc + cursubdim;
        
        clear projmat V;        
    end
    
    showinfo('   Learning Combined PCA ...', opts);
    
    % learn the combined model
    covcomb = slcov(intvecs, opts.weights, 0);
    [combevals, evecs] = slsymeig(covcomb);
    
    showinfo('   Truncating Combined PCA ...', opts);
    
    % truncate
    rk = decide_rank(combevals, (opts.er1 * energy.total) / energy.intpreserved);
    combevals = combevals(1:rk);
    combproj = evecs(:,1:rk);
    slwritearray(combproj, combprojfile); 
    
    energy.combpreserved = sum(combevals);
    diminfo.feadim = rk;
    
else
    energy.combpreserved = energy.intpreserved;
    diminfo.feadim = diminfo.intdim;
    
    combevals = [];   
    slignorevars(combevals);
end


%% Output

showinfo('Outputing Core file ...', opts);

corefilename = [filepath, '.mat'];

% change the inner file paths to relative path
dstdir = fileparts(filepath);
if isempty(dstdir) 
    pathprelen = 0;
elseif dstdir(end) ~= '\'
    pathprelen = length(dstdir) + 1;
else
    pathprelen = length(dstdir);
end

if pathprelen > 0
    for ib = 1 : NBlks
        fn = projfiles{ib};
        fn = fn(pathprelen+1:end);
        projfiles{ib} = fn;
    end
    if ~isempty(combprojfile)
        combprojfile = combprojfile(pathprelen+1:end);
    end
end    

slignorevars(combprojfile);


save(corefilename, ...
    'parstruct', ...
    'blocks', ...
    'projfiles', ...
    'combprojfile', ...
    'meanarr', ...
    'energy', ...
    'diminfo', ...
    'evalset', ...
    'combevals', ...
    '-v6');




%% Core Computing functions

function C = compute_localcov(data, meanarr, rangecell, d, n, w)

localmean = meanarr(rangecell{:});
localmean = reshape(localmean, [d, 1]);

if isnumeric(data)
    
    localarr = data(rangecell{:}, :);
    localarr = reshape(localarr, [d, n]);
    
    C = slcov(localarr, w, localmean);
    
else
    
    C = zeros(d, d);
    nfiles = length(data);
    
    cf = 0;
    for i = 1 : nfiles
        localarr = slreadarray(data{i});
        curn = size(localarr, length(rangecell)+1);
        
        if isempty(w)
            curw = [];
            tw = curn;
        else
            curw = w(cf+1:cf+curn);
            tw = sum(curw);
        end
        
        curcov = compute_localcov(localarr, meanarr, rangecell, d, curn, curw);        
        C = C + curcov * tw;
        
        cf = cf + curn;
    end
    
    if cf ~= n
        error('sltoolbox:sizmismatch', ...
            'The total number of units in the set of array files is not n');
    end  
    
    if isempty(w)
        C = C / n;
    else
        C = C / sum(w);
    end
    
end


function V = generate_pvecs(data, rangecell, oridim, subdim, n, meanarr, projmat)

if isnumeric(data)

    localarr = data(rangecell{:}, :);
    localarr = reshape(localarr, [oridim, n]);
    localmean = meanarr(rangecell{:});
    localmean = reshape(localmean, [oridim, 1]);
    D = sladdvec(localarr, -localmean, 1);    
    clear localmean localarr;
    V = projmat' * D;
    
    if size(V, 1) ~= subdim
        error('sltoolbox:sizmismatch', ...
            'Inconsistent sub dimension');
    end
        
else

    V = zeros(subdim, n);
    nfiles = length(data);
    cf = 0;
    for i = 1 : nfiles
        curdata = slreadarray(data{i});
        curn = size(curdata, length(rangecell) + 1);
        curV = generate_pvecs(curdata, rangecell, oridim, subdim, curn, meanarr, projmat);
        clear curdata;
        V(:, cf+1:cf+curn) = curV;
        cf = cf + curn;
    end
    
    if cf ~= n
        error('sltoolbox:sizmismatch', ...
            'The total number of units in the set of array files is not n');
    end  
end



%%  Auxiliary Functions

function check_valuerange(var, name, minval, maxval)

if var < minval || var > maxval
    error('sltoolbox:outofrange', ...
        'The variable %s should be between %f and %f', ...
        name, minval, maxval);
end


function str = generate_indexstring(nums, inds)

d = length(nums);
if any(inds <= 0) || any(inds > nums)
    error('The indices are beyond boundary');
end

% generate pattern
pats = cell(1, d);
for i = 1 : d
    curlen = length(int2str(nums(i)));
    pats{i} = sprintf('%%0%dd.', curlen);
end

% generate sub-strings
sstrs = cell(1, d);
for i = 1 : d    
    sstrs{i} = sprintf(pats{i}, inds(i));
end

% concatenate sub-strings
str = strcat(sstrs{:});
str(end) = [];   % delete the trailing point


function rk = decide_rank(evals, er)

cumevs = cumsum(evals);
rk = min(sum(cumevs < sum(evals) * er) + 1, length(evals));


function showinfo(message, opts)

if opts.verbose
    disp(message);
end



