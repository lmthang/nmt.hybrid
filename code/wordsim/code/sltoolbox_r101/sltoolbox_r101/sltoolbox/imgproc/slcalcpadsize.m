function [paddedsiz, roi, bmg] = slcalcpadsize(varargin)
%SLCALCPADSIZE Calculates the size of padding
%
% $ Syntax $
%   - [paddedsiz, roi, bmg] = slcalcpadsize(imgsize, bmg0)
%   - [paddedsiz, roi, bmg] = slcalcpadsize(imgsize, roi0, bmg0)
%
% $ Arguments $
%   - imgsize:      The whole size of the image
%   - roi0:         The rectangle of the target region 
%                   in the form of [t, b, l, m]
%   - bmg0:         The required boundary margins for the target region
%                   in the form [tm, bm, lm, rm]
%   - paddedsize:   The size of the padded image
%   - roi:          The rectangle of the target region in the padded image
%   - bmg:          The boundary margins for padding on the whole image
%
% $ Description $
%   - [paddedsiz, roi, bmg] = slcalcpadsize(imgsize, bmg0) calculates
%     the padding size when the target region is the whole image.
%
%   - [paddedsiz, roi, bmg] = slcalcpadsize(imgsize, roi0, bmg0) calculates
%     the padding size with the target region explicitly specified.
%
% $ History $
%   - Created by Dahua Lin, on Sep 1st, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slcalcpadsize', 2);
end

imgsize = varargin{1};
h0 = imgsize(1); w0 = imgsize(2);
if nargin == 2 
    roi0 = [1, h0, 1, w0];
    bmg0 = varargin{2};
else 
    if isempty(varargin{2})
        roi0 = [1, h0, 1, w0];
    else
        roi0 = varargin{2};
    end
    bmg0 = varargin{3};
end

%% compute padding boundary

[t0, b0, l0, r0] = sltakeval(roi0);
inner_mgs = [t0 - 1, h0 - b0, l0 - 1, w0 - r0];

bmg = max(bmg0 - inner_mgs, 0);

%% compute padded size

ph = h0 + bmg(1) + bmg(2);
pw = w0 + bmg(3) + bmg(4);
paddedsiz = [ph, pw];

%% compute roi in padded image

roi = [bmg(1) + t0, bmg(1) + b0, bmg(3) + l0, bmg(3) + r0];

