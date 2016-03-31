function D = sldiff_pw(X1, X2, type)
%SLDIFF_PW Measures the pair-wise difference
%
% $ Synatx $
%   - D = sldiff_pw(X1, X2, type)
%
% $ Arguments $
%   - X1, X2:             the two sample matrices
%   - type:               the type of difference measurement
%                         default = 'abssum'
%
% $ Description $
%   - D = sldiff_pw(X1, X2, type) computes the measurment of differences
%     between the samples in X1 and those in X2 in a pairwise manner.
%     All samples should be stored in columns. And the samples in X1 and 
%     X2 should be of the same dimension. If X1 and X2 are of sizes
%     dxn1 and dxn2 respectively. Then D is a matrix of size n1 x n2.
%
%   - The measurment types supported are listed below
%     \*
%     \t  Table 1. The difference measurement types                 \\
%     \h    name      &         description                         \\
%          'abssum'   &   sum of absolute values of differences     \\
%          'maxdiff'  &   maximum of absolute values of differences \\
%          'mindiff'  &   minimum of absolute values of differences \\
%     \*
%
% $ History $
%   - Created by Dahua Lin on Dec 06th, 2005
%   - Modified by Dahua Lin on Sep 10th, 2006
%       - Re-implement the core in C++: pwdiff_core
%       - The efficiency is increased by 10 times.
% 

%% parse and verify input arguments

if nargin < 3 || isempty(type)
    type = 'abssum';
end

switch type
    case 'abssum'
        pdmcode = 1;
    case 'maxdiff'
        pdmcode = 2;
    case 'mindiff'
        pdmcode = 3;
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid type of pwdiff computation: %s', type);
end

%% Compute

D = pwdiff_core(X1, X2, pdmcode);


    
    
    