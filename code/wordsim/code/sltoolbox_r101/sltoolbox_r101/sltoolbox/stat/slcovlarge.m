function C = slcovlarge(X, w, vmean, cachesize)
%SLCOVLARGE Computes large covariance matrix using memory-efficient way
%
% $ Syntax $
%   - C = slcovlarge(X, w, vmean, cachesize)
%
% $ Arguments $
%   - X:            the sample matrix
%   - w:            the weights of samples (default = [])
%                   if it is specifed, it should be a 1 x n vector
%   - vmean:        the pre-computed mean (default = [])
%   - cachesize:    the size of working cache (in the unit of Mbytes)
%
% $ Description $
%   - C = slcovlarge(X, w, vmean, cachesize) computes the large covariance 
%     matrix within a memory-limited context. Compared to slcov, which is
%     faster at the expense of using more memory, it is slower, however
%     the memory used is under control. Thus it can effectively prevent
%     from the situation of out of memory. 
%
% $ Remarks $
%   - The memory increased in the function will not exceed 
%     the size of the covariance matrix plus the cachesize.
%
%   - For vmean
%       - if it is empty, then a mean vector will be calculated
%       - if it is zero, then the data is assumed to be centralized
%       - if it is specified, then its size should be d x 1.
%
%   - The memory estimated to be used for each section is
%     if has no weight
%           2 x sd x sn + 2 x (sd + sn). 
%     else
%           3 x sd x sn + 2 x (sd + sn).
%     end
%     Here sd and sn are the sub-dimension
%     and sub-number of that section.
%
% $ History $
%   - Created by Dahua Lin, on Aug 17, 2006
%

%% parse and verify input arguments

if nargin < 4
    raise_lackinput('slcovlarge', 4);
end

if ndims(X) ~= 2 || ~isnumeric(X)
    error('sltoolbox:invalidarg', ...
        'X should be a 2D numeric matrix');
end

[d, n] = size(X);

if ~isempty(w)
    if ~isequal(size(w), [1, n])
        error('sltoolbox:sizmismatch', ...
            'The w should be an 1 x n vector');
    end
end

% convert cache size
cacheelems = floor(cachesize * 1e6 / 8);

if isempty(vmean)
    if cacheelems < d
        error('sltoolbox:notenoughmem', ...
            'The cache size is not enough to hold even a mean vector');
    end    
    cacheelems = cacheelems - d;
    vmean = slmean(X, w);
else
    if ~isequal(vmean, 0)
        if ~isequal(size(vmean), [d, 1])
            error('sltoolbox:sizmismatch', ...
                'The vmean should be a d x 1 vector');
        end
    end
end


%% decide partition struct

if cacheelems > 2*d*n + 2*(d+n)   % can compute without partitioning
    ps = [];            
else     
    if isempty(w)
        cblks = 2;
    else
        cblks = 3;
    end
            
    sd = floor( (cacheelems - 2 * n) / (cblks * n + 2) );
    divide_row = false;
    if sd <= 0
        sd = 1;
        if cacheelems < cblks + 4
            error('sltoolbox:notenoughmem', ...
                'The cache is not large enough');
        end
        sn = floor((cacheelems - 2) / (cblks + 2));
        divide_row = true;
    end
            
    ps = slpartition(d, 'maxblksize', sd);
    if divide_row
        rps = slpartition(d, 'maxblksize', sn);
    end        
end


%% compute

if isempty(ps)
    
    if ~isempty(w);
        X = weight_x(X, w);
    end
    X = shift_x(X, vmean);
    Xt = X';
    C = X * Xt;
    
else
    
    if ~divide_row      % each row is taken as integral
        
        C = zeros(d, d);        
        nsecs = length(ps.sinds);
        
        for i = 1 : nsecs
            for j = 1 : nsecs
                si = ps.sinds(i);
                ei = ps.einds(i);
                sj = ps.sinds(j);
                ej = ps.einds(j);
                                   
                if isempty(w)
                    curXj = X(sj:ej, :);  
                    curXj = shift_x(curXj, vmean, sj, ej);
                    curXjt = curXj';
                    clear curXj;
                    
                    curXi = X(si:ei, :);    
                    curXi = shift_x(curXi, vmean, si, ei);
                else
                    curXj = X(sj:ej, :);
                    curXj = shift_x(curXj, vmean, sj, ej);               
                    curXj = weight_x(curXj, w);
                    clear curwj;
                    curXjt = curXj';
                    clear curXj;
                    
                    curXi = X(si:ei, :);
                    curXi = shift_x(curXi, vmean, si, ei);
                    curXi = weight_x(curXi, w);
                    clear curwi;
                end
                
                C(si:ei, sj:ej) = curXi * curXjt;

            end
        end
                                
    else                % even each row need to be divided        
        
        slignorevars(rps);
        
        error('sltoolbox:rterror', ...
            'In current implementation, row-division is not implemeted yet');        
    end    
    
end
              

%% scale down

if isempty(w)
    C = C / n;
else
    tw = sum(w);
    C = C / tw;
end


%% Core computing function

function sx = weight_x(sx, w)

sn = size(sx, 2);
for i = 1 : sn
    sx(:, i) = sx(:, i) * sqrt(w(i));
end

function sy = shift_x(sx, vmean, sidx, eidx)

[sd, sn] = size(sx);
if ~isequal(vmean, 0)
    
    if nargin == 2
        curmean = vmean;
    else
        curmean = vmean(sidx:eidx);
    end    
    
    sy = zeros(sd, sn);
    for i = 1 : sn
        sy(:,i) = sx(:,i) - curmean;
    end
else
    sy = sx;
end




        

        