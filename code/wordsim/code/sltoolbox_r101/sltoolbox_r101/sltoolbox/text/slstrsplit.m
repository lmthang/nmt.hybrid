function strs = slstrsplit(srcstr, delimiters)
%SLSTRSPLIT splits a string into cell array of strings by delimiters
%
% $ Syntax $
%   - strs = slstrsplit(srcstr, delimiters)
%
% $ Arguments $
%   - srcstr:       the source string
%   - delimiters:   the array of delimiting chars
%
% $ Description $
%   - strs = slstrsplit(srcstr, delimiters) splits the source string into
%     a cell array of parts, which are delimited by the chars in
%     delimiters. 
% 
% $ Remarks $
%   - If for adjacent delimiters, the between will not will extracted.
%   - No further processing is applied, you can use functions like 
%     slcompresstext to achieve these goals.
% 
% $ History $
%   - Created by Dahua Lin, on Aug 13, 2006
%

%% determine delimiter positions

n0 = length(srcstr);
is_delimiter = false(n0, 1);

nd = length(delimiters);
for i = 1 : nd
    ch = delimiters(i);
    is_delimiter(srcstr == ch) = true;
end

dps = find(is_delimiter);
dps = dps(:)';

%% extract parts

if isempty(dps)
    strs = {srcstr};
else
    sps = [1, dps+1];
    eps = [dps-1, n0];
    fv = find(sps <= eps);
    if ~isempty(fv)    
        sps = sps(fv);
        eps = eps(fv);
        np = length(fv);
        strs = cell(np, 1);
        for i = 1 : np
            strs{i} = srcstr(sps(i):eps(i));
        end        
    else
        strs = {};
    end
end

        


