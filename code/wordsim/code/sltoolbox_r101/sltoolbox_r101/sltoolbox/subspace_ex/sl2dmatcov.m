function varargout = sl2dmatcov(type, data, matsiz, n, meanmat, PL, PR, w)
%SL2DMATCOV Computes the 2D matrix-covariances
%
% $ Syntax $
%   - CL = sl2dmatcov('CL', data, matsiz, n, meanmat, PL, PR, w)
%   - CR = sl2dmatcov('CR', data, matsiz, n, meanmat, PL, PR, w)
%   - [CL, CR] = sl2dmatcov('Both', data, matsiz, n, meanmat, PL, PR, w)
%
% $ Arguments $
%   - data:     the stack of matrices or the cell array of filenames of the
%               array files storing the matrices.
%   - matsiz:   the size of each matrix
%   - n:        the number of samples
%   - PL:       the left-projection matrix, default = []
%   - PR:       the right-projection matrix, default = []
%   - w:        the weights of the matrix samples, default = []
%
% $ Remarks $
%   - The function computes the 2D matrix-covariances according to 
%     the following formulas:
%       Y_i = PL^T * (X - meanX)* PR 
%       CL = ( sum_{i=1}^n w(i) * Y_i * Y_i^T ) / ( sum_{i=1}^n w(i) )
%       CR = ( sum_{i=1}^n w(i) * Y_i^T * Y_i ) / ( sum_{i=1}^n w(i) )
%     Following special cases are considered:
%       If w is empty, then we take w(i) = 1 for all i
%       If PL is empty, then we take PL as an identity matrix
%       If PR is empty, then we take PR as an identity matrix
%       If meanmat is 0, then we take meanmat as a zero matrix
%
% $ History $
%   - Created by Dahua Lin, on Jul 31st, 2006
%

%% parse and verify input arguments

if nargin < 5
    raise_lackinput('sl2dmatcov', 5);
end
if nargin < 6
    PL = [];
end
if nargin < 7
    PR = [];
end
if nargin < 8
    w = [];
end

matsiz = matsiz(:)';
if length(matsiz) ~= 2
    error('sltoolbox:invalidarg', ...
        'matsiz shoudl be a 2-elem vector');
end
if ~isequal(size(meanmat), matsiz)
    error('sltoolbox:invalidarg', ...
        'The size of mean matrix is not as specified');
end

d1 = matsiz(1);
d2 = matsiz(2);
if ~isempty(PL) && size(PL, 1) ~= d1
    error('sltoolbox:invalidarg', ...
        'The size of PL is inconsistent with the matrix dimension');
end
if ~isempty(PR) && size(PR, 1) ~= d2
    error('sltoolbox:invalidarg', ...
        'The size of PR is inconsistent with the matrix dimension');
end

if ~isempty(w)
    if length(w) ~= n
        error('The weights length is inconsistent with the number of samples');
    end
    tw = sum(w);
else
    tw = n;
end


%% Main body

if isnumeric(data)

    if ~isequal(size(data), [matsiz, n])
        error('sltoolbox:invalidarg', ...
            'The size of data is inconsistent with specified');
    end
    
    Y = compute_Y(data, meanmat, PL, PR);
    
    switch type
        case 'CL'
            CL = compute_SL(Y, w);
            CL = CL / tw;
            varargout = {CL};
        case 'CR'
            CR = compute_SR(Y, w);
            CR = CR / tw;
            varargout = {CR};
        case 'Both'
            [CL, CR] = compute_SLSR(Y, w);
            CL = CL / tw;
            CR = CR / tw;
            varargout = {CL, CR};
        otherwise
            error('sltoolbox:invalidarg', ...
                'invalid type: %s', type);
    end            
    
    
elseif iscell(data)
    
    nfiles = length(data);
    cf = 0;
    
    switch type
        case 'CL'
            for i = 1 : nfiles
                curarr = slreadarray(data{i});
                curn = size(curarr, 3);
                Y = compute_Y(curarr, meanmat, PL, PR);
                curCL = compute_SL(Y, w); 
                if i == 1
                    CL = curCL;
                else
                    CL = CL + curCL;
                end
                clear Y curCL;
                cf = cf + curn;
            end
            CL = CL / tw;
            varargout = {CL};
            
        case 'CR'
            for i = 1 : nfiles
                curarr = slreadarray(data{i});
                curn = size(curarr, 3);
                Y = compute_Y(curarr, meanmat, PL, PR);
                curCR = compute_SR(Y, w); 
                if i == 1
                    CR = curCR;
                else
                    CR = CR + curCR;
                end
                clear Y curCR;
                cf = cf + curn;
            end
            CR = CR / tw;
            varargout = {CR};
            
        case 'Both'
            for i = 1 : nfiles
                curarr = slreadarray(data{i});
                curn = size(curarr, 3);
                Y = compute_Y(curarr, meanmat, PL, PR);
                [curCL, curCR] = compute_SLSR(Y, w); 
                if i == 1
                    CL = curCL;
                    CR = curCR;
                else
                    CL = CL + curCL;
                    CR = CR + curCR;
                end
                clear Y curCL curCR;
                cf = cf + curn;
            end
            CL = CL / tw;
            CR = CR / tw;
            varargout = {CL, CR};
            
        otherwise
            error('sltoolbox:invalidarg', ...
                'invalid type: %s', type);        
    end
    
    if cf ~= n
        error('sltoolbox:sizmismatch', ...
            'The total number of samples is not n');
    end
            
else
    error('sltoolbox:invalidarg', ...
        'data should be a numeric array or a cell array of filenames');    
end



%% Core routine

function Y = compute_Y(X, M, PL, PR)

n = size(X, 3);
if isempty(PL)
    if isempty(PR)
        if isequal(M, 0)
            Y = X;
        else
            Y = zeros(size(X, 1), size(X, 2), n);
            for i = 1 : n
                Y(:,:,i) = X(:,:,i) - M;
            end
        end
    else
        Y = zeros(size(X, 1), size(PR, 2), n);
        if isequal(M, 0)           
            for i = 1 : n
                Y(:,:,i) = X(:,:,i) * PR;
            end
        else
            for i = 1 : n
                Y(:,:,i) = (X(:,:,i) - M) * PR;
            end
        end
    end
else
    PLT = PL';
    if isempty(PR)
        Y = zeros(size(PL, 2), size(X, 2), n);
        if isequal(M, 0)
            for i = 1 : n
                Y(:,:,i) = PLT * X(:,:,i);
            end
        else
            for i = 1 : n
                Y(:,:,i) = PLT * (X(:,:,i) - M);
            end
        end
    else
        Y = zeros(size(PL, 2), size(PR, 2), n);
        if isequal(M, 0)
            for i = 1 : n
                Y(:,:,i) = PLT * X(:,:,i) * PR;
            end
        else
            for i = 1 : n
                Y(:,:,i) = PLT * (X(:,:,i) - M) * PR;
            end
        end
    end
end


function SL = compute_SL(Y, w)

SL = zeros(size(Y, 1));
n = size(Y, 3);
if isempty(w)
    for i = 1 : n
        curY = Y(:, :, i);
        SL = SL + curY * curY';
    end
else
    for i = 1 : n
        curY = Y(:, :, i);
        SL = SL + w(i) * curY * curY';
    end
end

function SR = compute_SR(Y, w)

SR = zeros(size(Y, 2));
n = size(Y, 3);
if isempty(w)
    for i = 1 : n
        curY = Y(:, :, i);
        SR = SR + curY' * curY;
    end
else
    for i = 1 : n
        curY = Y(:, :, i);
        SR = SR + w(i) * curY' * curY;
    end
end    

function [SL, SR] = compute_SLSR(Y, w)

SL = zeros(size(Y, 1));
SR = zeros(size(Y, 2));
n = size(Y, 3);
if isempty(w)
    for i = 1 : n
        curY = Y(:, :, i);
        SL = SL + curY * curY';
        SR = SR + curY' * curY;
    end
else
    for i = 1 : n
        curY = Y(:, :, i);
        SL = SL + w(i) * curY * curY';
        SR = SR + w(i) * curY' * curY;
    end
end
    
        
