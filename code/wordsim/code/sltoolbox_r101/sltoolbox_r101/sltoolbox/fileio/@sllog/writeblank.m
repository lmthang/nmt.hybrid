function writeblank(logger)
%WRITEBLANK Writes a blank line to log
%
% $ Syntax $
%   - writeblank(logger)
%
% $ Description $
%   - writeblank(logger) writes one blank line to logging output.
%
% $ History $
%   - Created by Dahua Lin, on Aug 13, 2006
%

if logger.winshow
    disp(' ');
end

if ~isempty(logger.files)
    
    F = logger.files;
    nf = length(F);
    for i = 1 : nf        
        if F(i).isactive
            fprintf(F(i).fid, ' \n');
        end        
    end
        
end