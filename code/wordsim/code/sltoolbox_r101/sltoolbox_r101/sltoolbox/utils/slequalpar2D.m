function ps = slequalpar2D(siz, maxblk)
%SLEQUALPAR Partition a 2D array with balances for width and height
%
% $ Syntax $
%   - ps = slequalpar2D(siz, maxblk)
%
% $ Arguments $
%   - siz:      The size of the 2D array to be partitioned
%   - maxblk:   The maximum number of elements in each block
%   - ps:       The partition structure
%
% $ Description $
%   - ps = slequalpar2D(siz, maxblk) partitions the 2D array into blocks
%     such that for each block the height and width should be as close
%     as possible.
%
% $ History $
%   - Created by Dahua Lin, on Sep 8th, 2006
%

m = siz(1);
n = siz(2);
ne = m * n;

if ne <= maxblk    % not partitioned
    ps = struct('sinds', {1, 1}, 'einds', {m, n});
else
    b = max(sqrt(maxblk), 1);
    if b <= m && b <= n
        bm = b;
        bn = b;
    elseif b > m
        bm = m;
        bn = maxblk / m;
    else
        bn = n;
        bm = maxblk / n;
    end
    nm = ceil(m / bm);
    nn = ceil(n / bn);    
    ps = slpartition(siz, 'numblks', [nm, nn]); 
end


    
    





