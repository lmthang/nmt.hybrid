function slpwcomp_blks(X1, X2, ps, dstpath, compfunc, varargin)
%SLPWCOMP_BLKS Computes pairwise value matrix
%
% $ Syntax $
%   - slpwcomp_blks(X1, X2, ps, dstpath, compfunc, ...)
%
% $ Arguments $
%   - X1:           the first source sample matrix
%   - X2:           the second source sample matrix
%   - ps:           the partition structure of the target value matrix
%   - dstpath:      the destination path
%   - compfunc:     the normal pairwise computing function
%   
% $ Description $
%   - slpwcomp_blks(X1, X2, ps, dstpath, compfunc, ...) computes the 
%     large scale pairwise value matrix in a block-wise way. Suppose
%     X1 has n1 columns, while X2 has n2 columns, it will produce a
%     n1 x n2 matrix, when both n1 and n2 are large, then the huge
%     value matrix may not be held entirely in the memory. The function
%     partition the value matrix into several blocks according to 
%     the partition structure specified in ps, which can be obtained 
%     by slpartition. Then for each block, the function will select
%     corresponding section of samples from X1 and X2, then compute the
%     corresponding value block and store it to an array file. 
%     The name of core output file is <dstname>.mat, so you need not
%     to add .mat when you input dstname. Please note that the core
%     mat file does not contain the actual value matrix, but some 
%     information for retrievaling them. There are following variables
%     in the mat file:
%       - 'parstruct':      the partition structure
%       - 'blocks':         the 2D cell array of block limits
%       - 'data':           the 2D cell array of filenames of array data
%       - 'matsize':        the whole size of the value matrix
%  
% $ Remarks $
%   - compfunc is the underlying pairwise value matrix computation
%     function. It has a form f(X1, X2, ...), when it inputs X1 and
%     X2 with n1 and n2 columns respectively, it outputs an n1 x n2
%     value matrix. It can be function name, inline function or
%     function handle.
%     
% $ History $
%   - Created by Dahua Lin, on Aug 8th, 2006
%

%% parse and verify input arguments
    
if nargin < 5
    raise_lackinput('slpwcomp_blks', 5);
end
if ~isnumeric(X1) || ndims(X1) ~= 2 || ~isnumeric(X2) || ndims(X2) ~= 2
    error('sltoolbox:invalidargs', ...
        'X1 and X2 should be 2D numeric matrices');
end
if ~isstruct(ps) || length(ps) ~= 2
    error('sltoolbox:invalidargs', ...
        'ps should be a struct array with 2 elements');
end

n1 = size(X1, 2);
n2 = size(X2, 2);

parstruct = ps;
matsize = [n1, n2];

slignorevars(matsize);

%% Prepare Blocks

blocks = slparblocks(parstruct);
[nrows, ncols] = size(blocks);
nblocks = nrows * ncols;

%% Prepare Filenames

len1 = length(int2str(n1));
len2 = length(int2str(n2));

dstcorepath = [dstpath, '.mat'];
dstdir = slfilepart(dstpath, 'parent');
dsttitle = slfilepart(dstpath, 'name');
arrnamepat = sprintf('%s.c%%0%dd-%%0%dd.r%%0%dd-%%0%dd.arr', dsttitle, len1, len1, len2, len2);

data = cell(nrows, ncols);
for i = 1 : nrows
    for j = 1 : ncols
        cb = blocks{i, j};
        data{i, j} = sprintf(arrnamepat, cb(1, 1), cb(2, 1), cb(1, 2), cb(2, 2));
    end
end

%% Compute and store (blockwise)

if ~isempty(dstdir) && ~exist(dstdir, 'dir')
    mkdir(dstdir);
end

for k = 1 : nblocks
    curblock = blocks{k};
    curpath = sladdpath(data{k}, dstdir);
    
    curX1 = X1(:, curblock(1,1):curblock(2,1));
    curX2 = X2(:, curblock(1,2):curblock(2,2));
    
    M = feval(compfunc, curX1, curX2, varargin{:});
    
    slwritearray(M, curpath);
end

%% Save core file
save(dstcorepath, 'parstruct', 'blocks', 'data', 'matsize', '-v6');


