function [R, pixinds] = slpixneighbors(img, filtersize, varargin)
%SLPIXNEIGHBORS Extracts the neighborhood of pixels from an image
%
% $ Syntax $
%   - R = slpixneighbors(img, filtersize, ...)
%   - [R, pixinds] = slpixneighbors(...)
%
% $ Argument $
%   - img:          The image
%   - filtersize:   the spec of filter size and center position
%   - R:            the extracted neighborhood
%   - pixinds:      the indices of the pixels (1 x n row vector)
%
% $ Description $ 
%   - R = slpixneighbors(img, filtersize, ...) extracts the neighborhoods 
%     of the pixels in range rgn of img. Here, img can be either a
%     single-channel or multi-channel image. Please refer to the function
%     slfiltersize for details of filtersize. There are two types of
%     output: 'cols' and 'rects'. In the form of 'cols', the size of R
%     would be d x n, here d = number of pixels in each neighborhood,
%     n = the number of neighborhoods. In the form of 'rects', the size
%     of R is ph x pw x n (single-channel) or ph x pw x k x n (multi)
%     here ph and pw are respectively the height and width of each 
%     neighborhood.
%     You can specify the following properties:
%       - 'output':     the form of output: 'cols'|'rects'
%                       default = 'cols';
%       - 'roi':        The region of interest, in the form of
%                       [top, bottom, left, right], or being the 
%                       following strings:
%                       - 'full': the whole image (default)
%                       - 'confined': all the pixels with its neighborhood
%                                     confined in the image.
%       - 'mask':       The mask of useful regions. The mask should be
%                       of the size h x w, and only pixels with 
%                       corresponding mask value > 0 will be used.
%                       (default = [], means all pixels are enabled)
%       - 'samplestep': The step of sampling, in form of [sx, sy].
%                       (default = [1, 1])
%       - 'pad':        The parameter of padding
%                       can be the padded values, or padding type.
%                       Please refer to slpadimg for details.
%                       default = 'replicate';
%                       
%  $ History $
%    - Created by Dahua Lin, on Sep 2nd, 2006
%    - Modified by Dahua Lin, on Sep 10, 2006
%       - replace sladd by sladdvec to increase efficiency
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slpixneighbors', 2);
end

[h0, w0, k] = size(img);

[filtsiz, bmg0] = slfiltersize(filtersize);

opts.output = 'cols';
opts.roi = 'full';
opts.mask = [];
opts.samplestep = [1, 1];
opts.pad = 'replicate';
opts = slparseprops(opts, varargin{:});

if ~ismember(opts.output, {'cols', 'rects'})
    error('sltoolbox:invalidarg', ...
        'Invalid output form: %s', opts.output);
end

if ischar(opts.roi)
    switch opts.roi
        case 'full'
            roi0 = [1, h0, 1, w0];
        case 'confined'
            roi0 = [1+bmg0(1), h0-bmg0(2), 1+bmg0(3), w0-bmg0(4)];
        otherwise
            error('sltoolbox:invalidarg', ...
                'Invalid ROI option: %s', opts.roi);
    end
else
    roi0 = opts.roi;
    if ~(isnumeric(roi0) && isvector(roi0) && length(roi0) == 4)
        error('sltoolbox:invalidarg', 'Invalid form of roi');
    end
end

mask = opts.mask;
if ~isempty(mask)
    if ~isequal(size(mask), [h0 w0])
        error('sltoolbox:invalidarg', ...
            'The size of mask is not consistent with the image size');
    end
end

ss = opts.samplestep;
if ~isvector(ss) || length(ss) ~= 2
    error('sltoolbox:invalidarg', ...
        'The sample step should be a length-2 vector');
end

%% Main

R = [];
pixinds = [];

% process ROI

if roi0(2) < roi0(1) || roi0(4) < roi0(3)
    return;
end

% padding

[psiz, roi, bmg] = slcalcpadsize([h0 w0], roi0, bmg0);
h = psiz(1); w = psiz(2);

if any(bmg > 0)
    img = slpadimg(img, bmg, opts.pad);
    if ~isempty(mask)
        mask = slpadimg(mask, bmg, false);
    end
end

% select indices

inds_i = (roi(1):ss(1):roi(2))';
ni = length(inds_i);
inds_j = roi(3):ss(2):roi(4);
nj = length(inds_j);
if ni <= 0 || nj <= 0;
    return;
end
I = inds_i(:, ones(1, nj));
J = inds_j(ones(ni, 1), :);
inds = sub2ind([h w], I, J);
clear I J;

if isempty(mask)
    inds = inds(:);
else
    if islogical(mask)
        inds = inds(mask(inds));
    else
        inds = inds(mask(inds) > 0);
    end
end
if isempty(inds) 
    return;
end

pixinds = inds';
clear inds;
n = length(pixinds);

% generate neighboring indices
[fh, fw, fcx, fcy] = sltakeval(filtsiz);
fs_i = (1-fcx:fh-fcx)';
fs_j = 1-fcy:fw-fcy;

fI = fs_i(:, ones(1, fw), ones(k,1));
fJ = fs_j(ones(fh, 1), :, ones(k,1));
if k == 1
    rel_inds = fI + h * fJ;
else
    fs_k = reshape(0:k-1, [1,1,k]);
    fK = fs_k(ones(fh,1), ones(fw,1), :);
    rel_inds = fI + h * fJ + (h*w) * fK;
end
    
d = fh * fw * k;
rel_inds = rel_inds(:);

indsmap = pixinds(ones(d,1), :);
indsmap = sladdvec(indsmap, rel_inds, 1);

% extract neighborhood values

R = img(indsmap);
clear indsmap;

if strcmp(opts.output, 'rects')
    if k == 1
        R = reshape(R, [fh, fw, n]);
    else
        R = reshape(R, [fh, fw, k, n]);
    end
end

% convert indices

if nargout >=2 && any(bmg > 0) 
    [I, J] = ind2sub([h w], pixinds);
    I = I - bmg(1);
    J = J - bmg(3);        
    pixinds = sub2ind([h0, w0], I, J);
end





