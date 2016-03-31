function [fs, bmg] = slfiltersize(fs0)
%SLFILTERSIZE Extracts information from filtersize
%
% $ Syntax $
%   - [fs, bmg] = slfiltersize(fs0)
%
% $ Arguments $
%   - fs0:      The input filter size
%   - fs:       The full filter size form
%   - bmg:      The boundary margins
%
% $ Description $
%   - [fs, bmg] = slfiltersize(fs0) restores the full form of the input
%     filtersize. In sltoolbox, filter size can be specified in either
%     of the following forms:
%     \*
%     \t    Table.  The forms of the filter size                    \\
%     \h     name      &           syntax                           \\
%            full      & [height, width, center_y, center_x]        \\
%            sizeonly  & [height, width]                           
%                        The center will be computed as:
%                        cy = floor((1 + h) / 2)
%                        cx = floor((1 + w) / 2)                    \\
%            lenonly   & [len]
%                        height = width = len
%     \*
%     bmg is the boundary margins in the form of 
%     [top_margin, bottom_margin, left_margin, right_margin]
%
% $ History $
%   - Created by Dahua Lin, on Sep 1st, 2006
%


%% parse filter size

if ~isvector(fs0)
    error('sltoolbox:invalidarg', ...
        'fs0 should be a vector');
end

switch length(fs0)
    case 1
        h = fs0;
        w = fs0;
        cy = floor((1+h)/2);
        cx = floor((1+w)/2);
    case 2
        cencoords = floor((1 + fs0) / 2);
        h = fs0(1);
        w = fs0(2);
        cy = cencoords(1);
        cx = cencoords(2);
    case 4
        h = fs0(1);
        w = fs0(2);
        cy = fs0(3);
        cx = fs0(4);
    otherwise
        error('sltoolbox:sizmismatch', ...
            'The length of fs0 is illegal');
end

fs = [h, w, cy, cx];

%% compute boundary margins

if nargout >= 2
    tm = cy - 1;
    bm = h - cy;
    lm = cx - 1;
    rm = w - cx;
    bmg = [tm, bm, lm, rm];
end



