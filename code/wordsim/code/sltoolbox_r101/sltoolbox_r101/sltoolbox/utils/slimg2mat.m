function M = slimg2mat(img)
%SLIMG2MAT Converts an image array to a double matrix
%
% $ Syntax $
%   - M = slimg2mat(img)
%
% % Arguments $
%   - img:      the image
%   - M:        the converted matrix
%
% $ Description $
%   - M = slimg2mat(img) converts a variety of image arrays to a double
%     2D matrix. In detail, the RGB image will be turned to a grayscale 
%     image, in addition, the value of uint8 or other integer types will
%     be converted to double value with range [0, 1] by im2double.
%
% $ History $
%   - Created by Dahua Lin, on Jul 25th, 2006
%


% color processing
d = ndims(img);
n = size(img, 3);
if d > 3 || n == 2 || n > 3
    error('sltoolbox:invalidimg', ...
        'The image should be 1-channel gray image or 2-channel RGB image');
end
if n == 3
    img = rgb2gray(img);
end

% type conversion
if isa(img, 'double')
    M = img;
else
    M = im2double(img);
end



