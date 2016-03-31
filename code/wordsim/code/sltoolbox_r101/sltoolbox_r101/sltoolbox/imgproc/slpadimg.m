function imgpadded = slpadimg(img, padsize, varargin)
%SLPADIMG Pads an image with boundary 
%
% $ Syntax $
%   - imgpadded = slpadimg(img, padsize, padval)
%   - imgpadded = slpadimg(img, padsize, padtype)
%
% $ Arguments $
%   - img:          The original input image
%   - padsize:      The boundary widths in [top, bottom, left, right]
%                   in addition, it can be in the other two forms:
%                   [len] => top = bottom = left = right = len
%                   [ey, ex] => top = bottom = ey, left = right = ex
%   - padval:       The padded values
%   - padtype:      The type of padding 
%                   'replicate' | 'symmetric' | 'circular'
%   - imgpadded:    The padded image
%
% $ Description $
%   - imgpadded = slpadimg(img, padsize, padval) pads the image with
%     constant values. For single-channel image, padval should be a
%     scalar. For multi-channle image, padval can be a scalar, which 
%     indicates to pad all channels using the same value, or an array
%     with the number of elements as the number of channels. Then 
%     different channels will be padded with corresponding element.
%
%   - imgpadded = slpadimg(img, padsize, padtype) pads the image with
%     specified scheme. Either of the 'replicate', 'circular' or 
%     'symmetric'.
%
% $ History $
%   - Created by Dahua Lin, on Sep 1st, 2006
%

%% parse and verify input

if nargin < 3
    raise_lackinput('slpadimg', 3);
end

% process padsize

if ~isvector(padsize)
    error('sltoolbox:invalidarg', ...
        'The padsize should be a vector');
end

if length(padsize) == 1
    padsize = padsize * ones(1, 4);
elseif length(padsize) == 2
    padsize = [padsize(1), padsize(1), padsize(2), padsize(2)];
elseif length(padsize) == 4
    padsize = padsize(:)';
else
    error('sltoolbox:invalidarg', ...
        'The length of padsize is illegal');
end    


% decide number of channels k
d = ndims(img);
if d == 2
    k = 1;
elseif d == 3
    k = size(img, 3);
else
    imgsiz = size(img);
    k = prod(imgsiz(3:end));
end
    
padparam = varargin{1};
if isnumeric(padparam) || islogical(padparam)
    padtype = 'constant';
    padval = padparam;
    if numel(padval) == 1
        if k > 1
            padval = padval * ones(k, 1);
        end
    elseif numel(padval) == k
        padval = padval(:);
    else
        error('sltoolbox:sizmismatch', ...
            'The size of padval is illegal');
    end
elseif ischar(padparam)
    padtype = padparam;
    if ~ismember(padtype, {'replicate', 'circular', 'symmetric'})
        error('sltoolbox:invalidarg', ...
            'Invalid padding type: %s', padtype);
    end
else
    error('sltoolbox:invalidarg', 'The padding parameters is invalid');
end

%% Main skeleton

switch padtype
    case 'constant'
        imgpadded = pad_constant(img, padsize, padval);
    case 'replicate'
        imgpadded = pad_replicate(img, padsize);
    case 'circular'
        imgpadded = pad_circular(img, padsize);
    case 'symmetric'
        imgpadded = pad_symmetric(img, padsize);
end

%% Reshape

if d > 3
    imgsiz(1) = size(imgpadded, 1);
    imgsiz(2) = size(imgpadded, 2);
    imgpadded = reshape(imgpadded, imgsiz);
end


%% Core functions

function imgdst = pad_constant(img, padsize, padval)

[h0, w0, k] = size(img);
hd = h0 + padsize(1) + padsize(2);
wd = w0 + padsize(3) + padsize(4);

% make the constant layer
padval = reshape(padval, [1 1 k]);
imgdst = padval(ones(hd,1), ones(wd,1), :);

% put in the target
i0 = padsize(1) + 1; i1 = padsize(1) + h0;
j0 = padsize(3) + 1; j1 = padsize(3) + w0;

imgdst(i0:i1, j0:j1, :) = img(:, :, :);


function imgdst = pad_replicate(img, padsize)

[tm, bm, lm, rm] = sltakeval(padsize);
h0 = size(img, 1);
w0 = size(img, 2);

inds_i = [ones(1, tm), 1:h0, ones(1, bm) * h0];
inds_j = [ones(1, lm), 1:w0, ones(1, rm) * w0];

imgdst = img(inds_i, inds_j, :);


function imgdst = pad_circular(img, padsize)

[tm, bm, lm, rm] = sltakeval(padsize);
h0 = size(img, 1);
w0 = size(img, 2);

inds_i = mod(-tm:h0+bm-1, h0) + 1;
inds_j = mod(-lm:w0+rm-1, w0) + 1;

imgdst = img(inds_i, inds_j, :);


function imgdst = pad_symmetric(img, padsize)

[tm, bm, lm, rm] = sltakeval(padsize);
h0 = size(img, 1);
w0 = size(img, 2);

sni = [1:h0, h0:-1:1];
snj = [1:w0, w0:-1:1];

inds_i = sni(mod(-tm:h0+bm-1, h0*2) + 1);
inds_j = snj(mod(-lm:w0+rm-1, w0*2) + 1);

imgdst = img(inds_i, inds_j, :);








        
    
    
    
    
    