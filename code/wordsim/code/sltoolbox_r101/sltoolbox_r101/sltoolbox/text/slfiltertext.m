function Tf = slfiltertext(T, f, varargin)
%SLFILTERTEXT Processes the lines of text
%
% $ Syntax $
%   - Tf = slfiltertext(T, f, ...)
%
% $ Arguments $
%   - T:        the text to be filtered
%   - f:        the filter function
%   - Tf        the filtered text
%
% $ Description $
%   - Tf = slfiltertext(T, f, ...) processes (filters) every line
%     of the text T, in form of cell array of strings, and generated the
%     cell array of processed text Tf. the filtering is run as
%           filtered_line = f(source_line, ...)
%
% $ Remarks $
%   - You can specify f as empty, then the original text will be simply
%     returned without processing
%
% $ History $
%   - Created by Dahua Lin, on Aug 9th, 2006
%

%% parse and verify arguments

if nargin < 2
    raise_lackinput('slfiltertext', 2);
end


%% filter
if ~isempty(T)
    if ~isempty(f)
        nlines = length(T);
        Tf = cell(nlines, 1);
        for i = 1 : nlines            
            curline = T{i};            
            procline = feval(f, curline, varargin{:});
            Tf{i} = procline;            
        end
    else
        Tf = T;
    end
else
    Tf = {};
end


    
