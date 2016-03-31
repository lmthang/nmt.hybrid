function vfb = slvecfilters(fb)
%SLVECFILTERS Vectorizes the filter band
%
% $ Syntax $
%   - vfb = slvecfilters(fb)
%
% $ Arguments $
%   - fb:       The filter band
%   - vfb:      The vectorized filter band
%   
% $ Description $
%   - vfb = slvecfilters(fb) vectorizes the filter band. A set of filter
%     band is an array of size (ph x pw x ...). In its vectorized form,
%     each filter is vectorized as a row vector. Then for a set of 
%     filter band with same size, vfb is a matrix of size k x d, here 
%     d = ph x pw, while k is the number of filters.
%   
% $ History $
%   - Created by Dahua Lin on Sep 2nd, 2006
%

fbsiz = size(fb);
if ndims(fb) == 2
    k = 1;
else
    k = prod(fbsiz(3:end));
end
d = prod(fbsiz(1:2));

vfb = reshape(fb, [d, k])';


    
    