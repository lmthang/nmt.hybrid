function [f, r] = slgetinterpkernel(kername)
%SLGETINTERPKERNEL Gets the interpolation kernel function
%
% $ Syntax $
%   - [f, r] = slgetinterpkernel(kername)
%   
% $ Arguments $
%   - kername:      The name of the interpolation kernel
%   - f:            The function handle to the kernel
%   - r:            The effective radius of the kernel
%
% $ Description $
%   - [f, r] = slgetinterpkernel(kername) gets the function handle to 
%     an interpolation kernel and the corresponding effective radius. 
%     The supported kernel include:
%       - 'nearest':         The nearest neighbor interpolation
%       - 'linear':          The linear interpolation
%       - 'cubic':           The cubic interpolation
%     For generality, the kername can also be a cell array as
%     {f, r}. Then the function directly extracts them to output.
%     Here are the formulas for the kernels:
%       - 'nearest':        f(x) = 1, when |x| <= 0.5
%                                  0, when |x| > 0.5 
%       - 'linear':         f(x) = 1 - |x|, when |x| <= 1
%                                  0,       when |x| > 1
%       - 'cubic':          f(x) = 1 - 2|x|^2 + |x|^3,        when |x| <= 1
%                                  4 - 8|x| + 5|x|^2 - |x|^3, when 1 < |x| <= 2
%                                  0,                         when |x| > 2
%
% $ Remarks $
%   - All the kernel functions are vectorized, so they all support
%     both scalar input and array input of any dimension.
%   - The kernel functions are legal only within the effective radius,
%     in the outside region, the produced values are undefined (not
%     necessary zero). Such a design is for efficiency, so it is the
%     invoker's responsibility to guarantee the input is in valid 
%     range.
%
% $ History $
%   - Created by Dahua Lin, on Sep 2nd, 2006
%

%% Main skeleton

if ischar(kername)
    switch kername
        case 'nearest'
            f = @(x) 1;
            r = 0.5;
        case 'linear'
            f = @(x) 1 - abs(x);
            r = 1;
        case 'cubic'
            f = @cubic_interpolant;
            r = 2;
        otherwise
            error('sltoolbox:invalidarg', ...
                'The interpolation kernel name is unsupported: %s', kername);
    end
elseif iscell(kername)
    f = kername{1};
    r = kername{2};
    if ~isa(f, 'funtion_handle')
        error('sltoolbox:invalidarg', ...
            'The interpolation kernel is invalid');
    end
else
    error('sltoolbox:invalidarg', ...
            'The interpolation kernel is invalid');
end
        


%% Core functions

function y = cubic_interpolant(x)

x = abs(x);
p1 = 1 - x .* x .* (2 - x);
p2 = 4 + x .* (-8 + x .* (5 - x));
b1 = x < 1;
b2 = true - b1;
y = p1 .* b1 + p2 .* b2;


        
        
        
        
        
