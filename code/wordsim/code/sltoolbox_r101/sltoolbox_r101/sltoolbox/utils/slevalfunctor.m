function varargout = slevalfunctor(functor, varargin)
%SLEVALFUNCTOR Evaluates a functor
%
% $ Syntax $
%   - [O1, O2, ...] = slevalfunctor(functor, I1, I2, ...)
%
% $ Description $
%   - [O1, O2, ...] = slevalfunctor(functor, I1, I2, ...) evaluates 
%     the functor. Here, a functor refers to a function that can be
%     invokable with parameters. A functor is typically used in a 
%     framework function that need to invoke other functions with
%     both the variables generated inside the framework and the
%     variables offered from external environment. 
%     If there is no external parameters, the functor can be
%     given in the form of a function name, function handle or
%     inline object; if there are external parameters the functor
%     can be given in form of a cell array with the first cell being
%     the callable function while the other cells containing the
%     external parameters.
%
% $ Remarks $
%   - For an empty functor, it simply does nothing and returns.
%
% $ History $
%   - Created by Dahua Lin, on Aug 30, 2006
%

if isempty(functor)
    return;
end

if iscell(functor)
    func = functor{1};
    if length(functor) > 1
        params = functor(2:end);
    else
        params = {};
    end
else
    func = functor;
    params = {};
end

if isempty(func)
    return;
end

if nargout == 0
    feval(func, varargin{:}, params{:});
else
    varargout = cell(1, nargout);
    [varargout{:}] = feval(func, varargin{:}, params{:});
end
    
    
