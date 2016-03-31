function [thr, fa, fr] = slgetroc(thrs, fars, frrs, item, itempara)
%SLGETROC Computes some point from ROC Curve
%
% $ Syntax $
%   - [thr, fa, fr] = slgetroc(thrs, fars, frrs, item, itempara)
%
% $ Arguments $
%   - thrs:         the sampled threshold values
%   - fars:         the false accept rates at the sampled thresholds
%   - frrs:         the false reject rates at the sampled thresholds
%   - item:         the items to be evaluated
%   - itempara:     the extra parameter for the item
%   - thr:          the threshold at which the specified point is reached
%   - fa:           the false accept rate at the selected point
%   - fr:           the false reject rate at the selected point
%   
% $ Description $
%   [thr, fa, fr] = slgetroc(thrs, fars, frrs, item, itempara) Computes 
%   a required point from the ROC curves specified by thrs, fars, and 
%   frrs. The requirement on the point is specified by item and itempara.
%   \*
%   \t   Table 1.  The Items of ROC Retrieval    \\
%   \h      name     &    description            \\
%          'ratio'   &  solves the point where fr / fa = itempara   \\
%          'fixfa'   &  solves the point where fa = itempara        \\
%          'fixfr'   &  solves the point where fr = itempara        \\
%          'fixth'   &  solves the point where threshold = itempara \\
%          'best'    &  finds the point where 
%                       itempara * fa + (1 - itempara) * fr attains min. \\
%   \*
%
% $ Remarks $
%   - To increase accuracy, inverse-interpolation technique is used.
%
% $ History $
%   - Created by Dahua Lin on Jun 10th, 2005
%   - Modified by Dahua Lin on May 1st, 2006
%     - Base on the sltoolbox v4
% 

%% Parse and Verify
if nargin < 4
    raise_lackinput('slgetroc', 5);
end
if ~isequal(size(thrs), size(fars)) || ~isequal(size(thrs), size(frrs))
    error('sltoolbox:sizmismatch', ...
        'The sizes of thrs, fars and frrs are not consistent');
end
thrs = thrs(:);
fars = fars(:);
frrs = frrs(:);

% preprocessing (make it strictly monotonical)
fars = make_mono(fars);
frrs = make_mono(frrs);

%% Compute
usemethod = 'linear';
switch item
    case 'ratio'
        ratios = frrs ./ max(fars, eps);
        if nargin < 5 || isempty(itempara)
            itempara = 1;
        end
        thr = interp1(ratios, thrs, itempara, usemethod);
    case 'fixfa'
        if nargin < 5 || isempty(itempara)
            itempara = 0.1;
        end
        thr = interp1(fars, thrs, itempara, usemethod);
    case 'fixfr'
        if nargin < 5 || isempty(itempara)
            itempara = 0.1;
        end
        thr = interp1(frrs, thrs, itempara, usemethod);
    case 'fixth'
        if nargin < 5 || isempty(itempara)
            error('You must specify the parameter for fixth item');
        end 
        thr = itempara;
    case 'best'
        if nargin < 5 || isempty(itempara)
            itempara = 0.5;
        end
        [mv, p] = min(itempara * fars + (1 - itempara) * frrs);
        slignorevars(mv);
        thr = thrs(p);
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid item %s for slgetroc', item);
end

fa = interp1(thrs, fars, thr, usemethod);
fr = interp1(thrs, frrs, thr, usemethod);


function f = make_mono(f)

df = diff(f);

if f(end) >= f(1)
    df = max(df, eps);
else
    df = min(df, -eps);
end
f = [f(1); f(1) + cumsum(df)];





