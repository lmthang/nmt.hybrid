function varargout = slinvoke(invoke_descr, varargin)
%SLINVOKE Invokes a function
%
% $ Syntax $
%   - [y1, y2, ..., yn] = slinvoke(invoke_descr, x1, x2, ..., xm)
%
% $ Arguments $
%   - invoke_descr:       the invoking descriptor
%   - x1, x2, ..., xm:    the input arguments
%   - y1, y2, ..., yn:    the output arguments
%
% $ Description $
%   - [y1, y2, ..., yn] = slinvoke(invoke_descr, placepos, x1, x2, ..., xm) 
%     invokes some function with a specified manner. The invoking descriptor
%     can be in following form:
%     1. a string representing the function name, a function handle or an 
%        inline object. In this case, no fixed argument is binded. The x1, 
%        x2, ..., xm are directly fed to the function.
%     2. a cell array like {func, var_pos, fa1, fa2, ... fak}. Here
%        func is the function name, function handle or inline object that
%        is to be invoked. var_pos is the positions of variable arguments.
%        fa1, fa2, ... are the fixed(binded) arguments. If var_pos is [],
%        the variable arguments will be put first.
%
% $ Examples $
%   - Invoke a function without fixed arguments
%     \{
%           slinvoke('plot', x, y);
%           y = slinvoke({'sin', []}, x);
%     \}
%
%   - Invoke a function with fixed arguments binded, the following pairs
%     of statements are equivalent.
%     \{
%           K = slinvoke({'strfind', 1, 'pattern'}, s);
%           K = strfind(s, 'pattern');
%
%           slinvoke({'plot', [1 2], 'r-'}, x, y);
%           plot(x, y, 'r-');
%
%           slinvoke({'plot', 2, x, 'b+'}, y);
%           plot(x, y, 'b+');
%
%           b = slinvoke({'isequal', [], 0}, x);
%           b = isequal(x, 0);
%     \}
%
% $ History $
%   - Created by Dahua Lin on Dec 28th, 2005
%


%% parse input arguments
if ~iscell(invoke_descr)
    func = invoke_descr;
    var_pos =[];
    fix_args = {};
else
    len_invoke_descr = length(invoke_descr);
    func = invoke_descr{1};
    if len_invoke_descr >= 2
        var_pos = invoke_descr{2};
    else
        var_pos = [];
    end
    if len_invoke_descr >= 3
        fix_args = invoke_descr(3:end);
    else
        fix_args = {};
    end
end

if isempty(var_pos) && ~isempty(varargin)
    var_pos = 1:length(varargin);
end
n_fix_args = length(fix_args);
n_var_args = length(var_pos);
if length(varargin) ~= n_var_args
    error('sltoolbox:argmismatch', ...
        'The variable input arguments do not match that described in invoking descriptor');
end


%% organize input arguments
if n_fix_args == 0 && n_var_args == 0
    if nargout == 0
        feval(func);
    else
        [varargout{1:nargout}] = feval(func);
    end
elseif n_var_args == 0
    if nargout == 0
        feval(func, fix_args{:});
    else
        [varargout{1:nargout}] = feval(func, fix_args{:});
    end
else
    n_args = max(n_fix_args + n_var_args, max(var_pos));
    input_args = cell(1, n_args);
    indicator_var = false(1, n_args);
    indicator_var(var_pos) = true; 
    fix_pos = find(~indicator_var);
    
    for i = 1 : n_var_args
        input_args{var_pos(i)} = varargin{i};
    end
    
    for i = 1 : n_fix_args
        input_args{fix_pos(i)} = fix_args{i};
    end
    
    if nargout == 0
        feval(func, input_args{:});
    else
        [varargout{1:nargout}] = feval(func, input_args{:});
    end
end


    



