function slshowtext(T)
%SLSHOWTEXT Displays the text
%
% $ Syntax $
%   - slshowtext(T)
%
% $ Description $
%   - slshowtext(T) shows the text in cell array of lines.
%
% $ History $
%   - Created by Dahua Lin, on Aug 9th, 2006
%

if ~isempty(T)
    nlines = length(T);
    for i = 1 : nlines
        curline = T{i};
        if isempty(curline);
            disp(' ');
        else
            disp(curline);
        end
    end
end
