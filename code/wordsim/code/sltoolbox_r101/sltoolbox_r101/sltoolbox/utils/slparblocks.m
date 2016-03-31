function blocks = slparblocks(ps)
%SLPARBLOCKS Gets the blocks from partition structure
%
% $ Syntax $
%   - blocks = slparblocks(ps)
%
% $ Arguments $
%   - ps:       the partition structure generated from slpartition
%   - blocks:   the cell array of block ranges
%
% $ Description $
%   - blocks = slparblocks(ps) gets the a cell array of blocks 
%     corresponding to the partition structure. If there are d dimensions
%     and m1, m2, ..., md partitions along each dimension. Then an
%     m1 x m2 x ... x md cell array will be returned, with each cell
%     containing a 2 x d array, in the form of 
%     [s1, s2, ..., sd; e1, e2, ..., ed]. It means that the block of
%     data will be extracted from an whole array A as
%     A(s1:e1, s2:e2, ..., sd:ed).
%
% $ History $
%   - Created by Dahua Lin, on Jul 29th, 2006
%


% calculate block numbers

d = length(ps);

blknums = zeros(1, d);
for i = 1 : d
    blknums(i) = length(ps(i).sinds);
end

% get block ranges

blkinds = slallsubinds(blknums);
NBlks = prod(blknums);
blkinds = reshape(blkinds, [d, NBlks]);

Smat = zeros(d, NBlks);
Emat = zeros(d, NBlks);
for i = 1 : d
    Smat(i, :) = ps(i).sinds(blkinds(i, 1:NBlks));
    Emat(i, :) = ps(i).einds(blkinds(i, 1:NBlks));
end


% convert from block range array to block cell array

Smat = reshape(Smat, [1, d*NBlks]);
Emat = reshape(Emat, [1, d*NBlks]);
Bmat = [Smat; Emat];
clear Smat Emat;

tempd = zeros(1, NBlks);
tempd(:) = d;
blocks = mat2cell(Bmat, 2, tempd);

if d == 1
    blocks = blocks(:);
else
    blocks = reshape(blocks, blknums);
end





