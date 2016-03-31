function rimgs = slresizeimg(imgs, newsiz, interpker)
%SLRESIZEIMG Resizes the images by interpolation
%
% $ Syntax $
%   - rimgs = slresizeimg(imgs, newsiz, interpker)
%
% $ Arguments $
%   - imgs:         The set of images
%   - newsiz:       The new image size to be resized to
%                   It can be in two forms:
%                   - [new_height, new_width]
%                   - ratio to the original size
%   - interpker:    The interpolation kernel (default = 'linear')
%   - rimgs:        The resized images
%
% $ Description $
%   - rimgs = slresizeimg(imgs, newsiz, interpker) resizes the image
%     to new size by interpolation. 
%
% $ Remarks $
%   - The implementation is based on slimginterp.
%
% $ History $
%   - Created by Dahua Lin, on Sep 3, 2006
%

%% parse and verify input

if nargin < 2
    raise_lackinput('slresizeimg', 2);
end
h0 = size(imgs, 1);
w0 = size(imgs, 2);

if isnumeric(newsiz)
    if length(newsiz) == 1
        h = newsiz * h0;
        w = newsiz * w0;
    elseif length(newsiz) == 2
        [h, w] = sltakeval(newsiz);
    else
        error('sltoolbox:invalidarg', 'The newsiz is invalid');
    end
else
    error('sltoolbox:invalidarg', 'The newsiz is invalid');
end

if nargin < 3 || isempty(interpker)
    interpker = 'linear';
end

%% generate coordinate map

I = linspace(1, h0, h)';
J = linspace(1, w0, w);
I = I(:, ones(1, w));
J = J(ones(h, 1), :);

%% Do interpolation

rimgs = slimginterp(imgs, I, J, interpker);


    