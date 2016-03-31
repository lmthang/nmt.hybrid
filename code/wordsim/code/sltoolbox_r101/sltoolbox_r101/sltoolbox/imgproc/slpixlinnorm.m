function dstimgs = slpixlinnorm(imgs, mu, sigma)
%SLPIXLINNORM Performs linear normalization on pixel values
%
% $ Syntax $
%   - dstimgs = slpixlinnorm(imgs);
%   - dstimgs = slpixlinnorm(imgs, mu, sigma)
%
% $ Arguments $
%   - imgs:     the array of images
%   - mu:       the mean pixel value to be normalized to (default = 0)
%   - sigma:    the standard deviation relative to mean pixel (default = 1)
%
% $ Description $
%   - dstimgs = slpixlinnorm(imgs, mu, sigma) performs linear normalization
%     on the image pixels so that the average pixel value is set to mu
%     while the standard deviation is set to sigma. The normalization is
%     conducted on each page(channel) respectively.
%
%   - dstimgs = slpixlinnorm(imgs) performs linear pixel value
%     normalization using default values.
%
% $ History $
%   - Created by Dahua Lin, on Aug 8th, 2006
%

%% parse and verify input arguments

if ~isa(imgs, 'double')
    imgs = im2double(imgs);
end
[h, w, n] = size(imgs);

if nargin < 2 || isempty(mu)
    mu = 0;
end

if nargin < 3 || isempty(sigma)
    sigma = 1;
end

d = h * w;

%% perform normalization

if n == 1
    dstimgs = normalize_page(imgs, d, mu, sigma);
else
    dstimgs = zeros(size(imgs));
    for i = 1 : n
        dstimgs(:,:,i) = normalize_page(imgs(:,:,i), d, mu, sigma);
    end
end


function dstimg = normalize_page(img, d, mu, sigma)

curimg = img(:);

% compute current mean value
cur_mv = sum(curimg) / d;

% shift to zero mean
curimg = curimg - cur_mv;

% compute current standard deviation
cur_std = norm(curimg) / sqrt(d);

% normalize to specified std dev
k = sigma / cur_std;
curimg = curimg * k;

% shift to specified mean
if mu ~= 0
    curimg = curimg + mu;
end

% reshape back to origin shape
dstimg = reshape(curimg, size(img));






