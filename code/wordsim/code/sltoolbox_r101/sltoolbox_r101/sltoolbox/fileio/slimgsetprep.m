function slimgsetprep(srcfolder, dstpath, matsize, maxsec)
%SLIMGSETPREP organizes the images in a MATLAB friendly way
%
% $ Syntax $
%   - slimgsetprep(srcfolder, dstpath, matsize, maxsec) 
%
% $ Argument $
%   - srcfolder:        the source file folder
%   - dstpath:          the destination file (without extension name)
%   - matsize:          the matrix size (row vector): [nrows, ncols]
%   - maxsec:           the maximum section size
%
% $ Description $
%   - slimgsetprep(srcfolder, dstpath, imgsiz, maxsec) prepares the matlab 
%     files in destination for a set of images. The requirement is
%       1) In source folder: there are all the images and a DSDML 
%          description file named dataset.xml
%       2) For destination, say abc, then there would be a core file named
%          abc.xml. And if the actual data is separately stored, there are
%          a series of files named abc.arr.<begin index>-<last index>. For
%          example, abc.arr.0001-2000, means that the 1st t0 2000th samples
%          are stored in the array file. 
%       3) maxsec is the number of samples in each section. If maxsec is
%          specified, the images are stored in separate array files. If 
%          maxsec is not specified or empty, the images are stored in core
%          file.
%       4) The core file is a MAT file with following variables:
%           'desc':         the DSDML data object descriptor
%           'sections':     the starting and ending indices of all sections
%           'matsize':      the row vector describing the size of images
%           'data':         the image array
%                           if imgs is numeric, it is the actual image
%                           array.
%                           if imgs is a cell array, it is the set of
%                           array filenames.
%   
% $ History $
%   - Created by Dahua Lin, on Jul 26th, 2006
%

%% Parse and verify input arguments

if nargin < 3
    raise_lackinput('slimgsetprep', 3);
end

isseparate = false;
if nargin >= 4
    isseparate = ~isempty(maxsec);
end

matsize = matsize(:)';
if length(matsize) ~= 2
    error('sltoolbox:invalidarg', ...
        'imgsize should be a 2-element vector');
end


%% Read the dataset

descrfile = [srcfolder, '\dataset.xml'];
desc = dataset(descrfile);

N = desc.numsamples;
fns = desc.filenames;

%% Process data

if ~isseparate
    
    datasiz = [matsize, N];
    data = zeros(datasiz);
    
    for i = 1 : N
        img = imread([srcfolder, '\', fns{i}]);
        img = slimg2mat(img);
        data(:,:,i) = img;
    end
        
else
    
    sections = slpartition(N, 'maxblksize', maxsec);    
    nsecs = length(sections.sinds);
    
    data = cell(nsecs, 1);    
    
    % process filenames
    dstdir = fileparts(dstpath);
    numlen = length(num2str(N));    
    if isempty(dstdir)  % local dst
        dstfnpat = [dstpath, sprintf('.arr.%%0%dd-%%0%dd', numlen, numlen)];
        dstfppre = [];
    else
        dstfnpat = [dstpath(length(dstdir)+2:end), sprintf('.arr.%%0%dd-%%0%dd', numlen, numlen)];
        dstfppre = [dstdir, '\'];
    end
    
    for i = 1 : nsecs        
        si = sections.sinds(i);
        ei = sections.einds(i);
        
        curfn = sprintf(dstfnpat, si, ei);
        curfp = [dstfppre, curfn];
        data{i} = curfn;
        
        curN = ei - si + 1;
        arr = zeros([matsize, curN]);
        
        for j = 1 : curN
            curidx = si + j - 1;
            img = imread([srcfolder, '\', fns{curidx}]);
            img = slimg2mat(img);        
            arr(:,:,j) = img;                                                
        end        
        
        slwritearray(arr, curfp);        
    end
        
end


%% Output core

corefile = [dstpath, '.mat'];
save(corefile, 'desc', 'sections', 'matsize', 'data', '-v6'); 




