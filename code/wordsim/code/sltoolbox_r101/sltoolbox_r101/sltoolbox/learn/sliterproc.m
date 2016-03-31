function [objects, info] = sliterproc(objects, iterfunctor, cmpfunctor, hasrecord, varargin)
%SLITERPROC Runs a general iterative process
%
% $ Syntax $
%   - objects = sliterproc(objects, iterfunctor, cmpfunctor, hasrecord, ...)
%   - [objects, info] = sliterproc(objects, iterfunctor, cmpfunctor, hasrecord, ...)
%
% $ Arguments $
%   - objects:      The models and data referred to in the process
%                   The input one is the initial objects
%                   The output one is the resulting objects.
%   - iterfunctor:  the functor invoked in each iteration, in the form:
%                   objects = f(objects, ...)
%                   If the process needs to be recorded, it is like:
%                   [objects, rec] = f(objects, ...)
%   - cmpfunctor:   The functor to compare two set of objects and 
%                   determine whether the process is converged.
%                   It is in the following form:
%                   is_converged = f(objects_prev, objects_current, ...)
%   - hasrecord:    whether the process is recorded
%   - info:         The struct of iteration information
%                   - converged:  whether the process has been converged
%                   - numiters:   the number of iterations
%                   - records:    the struct array of the records
%                     (this field exists when the process is recorded)
%   
% $ Description $
%   - objects = sliterproc(objects, iterfunctor, cmpfunctor, ...) runs
%     a specified iterative process on a set of models and data. 
%     You can specify the following properties to control the iteration:
%     \*
%     \t    Table.  Iteration Process Control Parameters
%     \h      name      &        description
%           'maxiter'   &  The maximum number of iterations 
%                          (default = inf)
%           'cvgcount'  &  How many continous converged iteration is 
%                          satisfied before it stops the whole process.
%                          (default = 1)
%           'verbose'   &  Whether to show the process of iteration     
%                          (default = true)
%           'titlebreak'&  Whether to break the line after displaying
%                          the iteration title. (default = true);
%     \*
%     The properties given above are called iteration control properties.
%     They are typically specified as a whole in a cell array in the
%     caller.
%
% $ History $
%   - Created by Dahua Lin on Aug 31, 2006
%

%% parse and verify input

if nargin < 4
    raise_lackinput('sliterproc', 4);
end

opts.maxiter = inf;
opts.cvgcount = 1;
opts.verbose = true;
opts.titlebreak = true;
opts = slparseprops(opts, varargin{:});

%% Main skeleton

slsharedisp_attach('sliterproc', 'show', opts.verbose);

niter = 0;
nconverged = 0;
while niter < opts.maxiter && nconverged < opts.cvgcount
    
    niter = niter + 1;
    
    if opts.titlebreak
        slsharedisp('Iteration %d', niter);
    else
        slsharedisp_word('Iteration %d: ', niter);
    end
    
    slsharedisp_incindent;
    
    % run iteration
    objects_prev = objects;
    if hasrecord
        [objects, records(niter, 1)] = slevalfunctor(iterfunctor, objects);
    else
        objects = slevalfunctor(iterfunctor, objects);
    end
    
    % compare and determine convergence
    isconverged = slevalfunctor(cmpfunctor, objects_prev, objects);
    if isconverged
        nconverged = nconverged + 1;
    else
        nconverged = 0;
    end        
    
    slsharedisp_decindent;
    
end

isconverged = (nconverged >= opts.cvgcount);
if isconverged
    slsharedisp('Iteration process converged');
end


slsharedisp_detach();

%% Output information

if nargout >= 2
    info.converged = isconverged;
    info.numiters = niter;
    if hasrecord
        info.records = records;
    end
end


