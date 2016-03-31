function fimgs = slapplyfilterband(imgs, filterband, filtersize, varargin)
%SLAPPLYFILTERBAND Applies filter band to filter images in batch
%
% $ Syntax $
%   - fimgs = slapplyfilterband(imgs, filterband, filtersize, ...)
%
% $ Arguments $
%   - imgs:         The images to be filtered
%   - filterband:   The set of filter band
%   - filtersize:   The spec of filter size
%   - fimgs:        The filtered images
%
% $ Description $
%   - fimgs = slapplyfilterband(imgs, filterband, filtersize, ...) applies 
%     a set of filterbands to images in batch. Suppose there are k filters, 
%     imgs is an array of size h0 x w0 x n1 x n2 x ... x nm, then fimgs is
%     an array of size h x w x n1 x n2 x ... x nm x k, here h and w are
%     respectively the height and width of the filtered image.
%     You can further specify the following properties:
%       - 'roi':    The ROI in original image (default = 'full')
%                   'full'|'confined'|[t, b, l, r]
%                   Please refer to slpixneighbors for details on ROI.
%       - 'fbform': The form of filterband (default = 'normal')
%                   - 'normal':  normal form
%                   - 'vec':     vectorized form
%     
% $ History $
%   - Created by Dahua Lin, on Sep 2nd, 2006
%

%% parse and verify input

if nargin < 2
    raise_lackinput('slapplyfilterband', 2);
end

imgsiz = size(imgs);

if ndims(imgs) == 2
    n = 1;
else
    n = prod(imgsiz(3:end));
end

[filtsiz, bmg] = slfiltersize(filtersize);
fh = filtsiz(1);
fw = filtsiz(2);


opts.roi = 'full';
opts.fbform = 'normal';
opts = slparseprops(opts, varargin{:});

if ischar(opts.roi)
    switch opts.roi
        case 'full'
            h = imgsiz(1); w = imgsiz(2);
        case 'confined'
            h = imgsiz(1) - bmg(1) - bmg(2);
            w = imgsiz(2) - bmg(3) - bmg(4);
        otherwise
            error('sltoolbox:invalidarg', ...
                'Invalid ROI option: %s', opts.roi);
    end
elseif isnumeric(opts.roi)
    h = opts.roi(2) - opts.roi(1) + 1;
    w = opts.roi(4) - opts.roi(3) + 1;
else
    error('sltoolbox:invalidarg', 'Invalid ROI parameter');
end

fimgs = [];
if h <= 0 || w <= 0
    return;
end

if ~ismember(opts.fbform, {'normal', 'vec'})
    error('sltoolbox:invalidarg', 'Invalid fbform: %s', opts.fbform);
end

%% Main

% prepare vectorized filterband

switch opts.fbform
    case 'normal'
        if size(filterband, 1) ~= fh || size(filterband, 2) ~= fw
            error('sltoolbox:sizmismatch', ...
                'The size of filterband is not consistent.');
        end
        vfb = slvecfilters(filterband);
    case 'vec'
        vfb = filterband;
end
k = size(vfb, 1);

% do filtering

if n == 1
    fimgs = filter_oneimg(imgs, opts.roi, vfb, filtersize, h, w, k);
else
    fsiz = [h, w, imgsiz(3:end), k];
    chsiz = [h, w, 1, k]; 
    fimgs = zeros(h, w, n, k);
    for i = 1 : n
        curimg = imgs(:,:,i);
        curf = filter_oneimg(curimg, opts.roi, vfb, filtersize, h, w, k);        
        fimgs(:,:,i,:) = reshape(curf, chsiz);
    end
    fimgs = reshape(fimgs, fsiz);
end



%% Core function

function fimg = filter_oneimg(img, roi, vfb, filtersize, h, w, k)
    
NBs = slpixneighbors(img, filtersize, 'roi', roi, 'output', 'cols');
fimg = vfb * NBs;  % k x npix
fimg = fimg';     % npix x k
fimg = reshape(fimg, [h, w, k]);



            








