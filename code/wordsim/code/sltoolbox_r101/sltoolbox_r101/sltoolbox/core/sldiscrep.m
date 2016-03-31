function d = sldiscrep(X1, X2, measure, r1)
%SLDISCREP Evaluates the discrepancy of two arrays
%
% $ Syntax $
%   - d = sldiscrep(X1, X2, measure)
%   - d = sldiscrep(X1, X2, measure, true)
%
% $ Arguments $
%   - X1:       the first array
%   - X2:       the second array
%   - measure:  the name of measure
%   - d:        the measure value
%
% $ Description $
%   - d = sldiscrep(X1, X2, measure) evaluates the discrepancy between
%     X1 and X2 according to the specified measure. 
%
%     \*
%     \t   Table 1. Discrepancy Measures
%     \h    name        &     description
%          'fro'        & Compute the Frobenius between two arrays
%                         that is to sum the square differences and
%                         compute the square root.
%          'avgfro'     & Compute the average Frobenius norm between
%                         two arrays, that is to average the square
%                         differences, and compute the square root.
%          'energy'     & Compute the total difference square energy.
%          'avgenergy'  & Compute the average difference square energy.
%          'maxdiffabs' & Compute the maximum element-wise absolute 
%                         difference
%          'maxdiffnrm' & Compute the maximum column-wise difference
%                         norm.
%     \*  
%
%   - d = sldiscrep(X1, X2, measure, true) computes the relative measure
%     with X1 as reference. By default, this mode is disabled.
%
% $ History $
%   - Created by Dahua Lin, on Aug 17, 2006
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('sldiscrep', 3);
end

if ~isequal(size(X1), size(X2))
    error('sltoolbox:sizmismatch', ...
        'The sizes of X1 and X2 are different');
end

if nargin < 4 || isempty(r1)
    r1 = false;
end

%% compute

switch measure
    case 'fro'
        D = X1(:) - X2(:);
        d = sqrt(sum(D.^2));
        clear D;
        if r1
            d0 = sqrt(sum(X1(:).^2));
            d = d / d0;
        end        
        
    case 'avgfro'
        n = numel(X1);
        D = X1(:) - X2(:);
        d = sqrt(sum(D.^2) / n);
        clear D;
        if r1
            d0 = sqrt(sum(X1(:).^2) / n);
            d = d / d0;
        end  
        
    case 'energy'
        D = X1(:) - X2(:);
        d = sum(D.^2);
        clear D;
        if r1
            d0 = sum(X1(:).^2);
            d = d / d0;
        end  
        
    case 'avgenergy'
        n = numel(X1);
        D = X1(:) - X2(:);
        d = sum(D.^2) / n;
        clear D;
        if r1
            d0 = sum(X1(:).^2) / n;
            d = d / d0;
        end  
        
    case 'maxdiffabs'
        if ~r1
            D = X1(:) - X2(:);
            d = max(abs(D));
        else
            D = (X1(:) - X2(:)) ./ X1(:);
            d = max(abs(D(:)));
        end
        
    case 'maxdiffnrm'
        if ~r1
            dn = slnorm(X1 - X2);
            d = max(dn(:));
        else
            dn = slnorm(X1 - X2);
            dn1 = slnorm(X1);
            d = max(dn(:) ./ dn1(:));
        end
        
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid measure name %s', measure);
end


    



