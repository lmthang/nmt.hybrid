function R = slcountlines(folderpath, fnreport)
%SLCOUNTLINES Count the lines of m-files in a folder and make a report
%
% $ Syntax $
%   - R = slcountlines(folderpath)
%   - slcountlines(folderpath, fnreport)
%   - R = slcountlines(folderpath, fnreport)
%
% $ Arguments $
%   - folderpath:       the path of the target folder 
%   - fnreport:         the filename of the report
%   - R:                the struct array storing the statistics     
%
% $ Description $
%   - R = slcountlines(folderpath) makes statistics on the source codes
%     in a matlab toolbox folder and returns the statistics by R.
%     R is a struct with following fields:
%       1. name:     the current level name of the folder
%       2. info:     the struct array recording the statistics, the
%                    statistics struct has following fields:
%                    a. filename
%                    b. length (number of bytes of the whole file)
%                    c. totallines (total number of source code lines)
%                    d. codelines
%                    e. commentlines
%                    f. emptylines
%       3. subdir:   the struct array of sub-folders
%       4. stat:     a struct representing the summarized statistics:
%                    a. numfiles
%                    b. totallines
%                    c. codelines
%                    d. commentlines
%                    e. emptylines
%   - slcountlines(folderpath, fnreport) makes statistics on the source
%     codes in a matlab toolbox folder and generates a report file.
%
%   - R = slcountlines(folderpath, fnreport) makes statistics on the 
%     source codes in a matlab toolbox folder, generates a report file,
%     and returns the struct.
%
% $ History $
%   - Created by Dahua Lin on May 2nd, 2006
%

%% parse and verify input arguments

if nargin >= 2
    report_required = true;
else
    report_required = false;
end


%% Carry out tasks

% make statistics

R = make_folder_stat(folderpath);

% make report

if report_required
    fh = fopen(fnreport, 'wt');    
    if fh > 0
        make_folder_report(fh, [], R);
        fclose(fh);
    else
        error('sltoolbox:ioerror', ...
            'Fail to create file %s for reporting', fnreport);
    end            
end



%% The recursive function for making folder-level statistics
function R = make_folder_stat(folderpath)


% initialize the struct
[pstr, foldername] = fileparts(folderpath);
slignorevars(pstr);
clear pstr;

R.name = foldername;
R.info = struct( ...
    'filename', {}, ...
    'length', {}, ...
    'totallines', {}, ...
    'codelines', {}, ...
    'commentlines', {}, ...
    'emptylines', {});

R.subdir = struct( ...
    'name', {}, ...
    'info', {}, ...
    'subdir', {}, ...
    'stat', {});

R.stat = struct( ...
    'numfiles', 0, ...
    'totallines', 0, ...
    'codelines', 0, ...
    'commentlines', 0, ...
    'emptylines', 0);

% list candidates
L = dir(folderpath);
nitems = length(L);
countdir = 0;
countfile = 0;

% process items
for i = 1 : nitems
    
    curitem = L(i);
    
    if curitem.isdir    % for a sub folder
        if ~strcmp(curitem.name, '.') && ~strcmp(curitem.name, '..') % non-trivial folder            
            cursub = make_folder_stat(fullfile(folderpath, curitem.name));                        
            if ~isempty(cursub.info) || ~isempty(cursub.subdir)     % non-empty folder
                countdir = countdir + 1;
                R.subdir(countdir) = cursub;
                
                R.stat.numfiles = R.stat.numfiles + cursub.stat.numfiles;
                R.stat.totallines = R.stat.totallines + cursub.stat.totallines;
                R.stat.codelines = R.stat.codelines + cursub.stat.codelines;
                R.stat.commentlines = R.stat.commentlines + cursub.stat.commentlines;
                R.stat.emptylines = R.stat.emptylines + cursub.stat.emptylines;
                
            end
        end                
    else                % for a file
        if length(curitem.name) > 2 && strcmpi(curitem.name(end-1:end), '.m') % an m-file           
            curinfo.filename = curitem.name;
            curinfo.length = curitem.bytes;
            curinfo = make_source_stat(fullfile(folderpath, curitem.name), curinfo);
            
            countfile = countfile + 1;                                   
            R.info(countfile) = curinfo;
            
            R.stat.numfiles = R.stat.numfiles + 1;
            R.stat.totallines = R.stat.totallines + curinfo.totallines;
            R.stat.codelines = R.stat.codelines + curinfo.codelines;
            R.stat.commentlines = R.stat.commentlines + curinfo.commentlines;
            R.stat.emptylines = R.stat.emptylines + curinfo.emptylines;
            
        end            
    end
    
end


%% The function for making source file-level statistics
function info = make_source_stat(filepath, info)

txt = slreadtext(filepath);
n = length(txt);

labels = zeros(n, 1);
% label meanings:
%   0 - empty line
%   1 - comment line
%   2 - code line

for i = 1 : n    
    curline = strtrim(txt{i});
    if ~isempty(curline)
        if curline(1) == '%'
            labels(i) = 1;
        else
            labels(i) = 2;
        end
    end    
end

info.totallines = n;
info.codelines = sum(labels == 2);
info.commentlines = sum(labels == 1);
info.emptylines = sum(labels == 0);


%% The function for making folder-level reports 
function make_folder_report(fh, parentpath, R)

curpath = fullfile(parentpath, R.name);

% print header
fprintf(fh, 'Folder: %s \r\n', curpath);
fprintf(fh, '------------------------------------------------------------\r\n');
fprintf(fh, '\n');

% print files
nfiles = length(R.info);
if nfiles > 0
    fprintf(fh, '%-25s \t %15s \t %15s \t %15s \t %15s \r\n', ...
        'filename', 'totallines', 'code', 'comment', 'empty');
    for i = 1 : nfiles
        make_file_reportline(fh, R.info(i)); 
    end
end
fprintf(fh, '\r\n');

% print sub-folders
nsubdirs = length(R.subdir);
for i = 1 : nsubdirs
    make_folder_report(fh, curpath, R.subdir(i));
end

% print summarized statistics
fprintf(fh, 'Summary: \n');
vinfo.filename = sprintf('total files %d', R.stat.numfiles);
vinfo.totallines = R.stat.totallines;
vinfo.codelines = R.stat.codelines;
vinfo.commentlines = R.stat.commentlines;
vinfo.emptylines = R.stat.emptylines;
make_file_reportline(fh, vinfo);

% print margin blank
fprintf(fh, '\r\n \r\n');


%% The function for making file-level report
function make_file_reportline(fh, info)

fprintf(fh, '%-25s \t %15d \t %15d \t %15d \t %15d \r\n', ...
    info.filename, ...
    info.totallines, ...
    info.codelines, ...
    info.commentlines, ...
    info.emptylines);
















