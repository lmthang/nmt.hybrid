function h = sldrawgraph(G, X, Xt, ppedges, ppsn, pptn)
%SLDRAWGRAPH Draws a graph
%
% $ Syntax $
%   - sldrawgraph(G, X)
%   - sldrawgraph(G, X, Xt)
%   - sldrawgraph(G, X, [], ppedges)
%   - sldrawgraph(G, X, [], ppedges, ppsn)
%   - sldrawgraph(G, X, Xt, ppedges)
%   - sldrawgraph(G, X, Xt, ppedges, pptn)
%   - h = sldrawgraph(...)
%
% $ Arguments $
%   - G:        The graph in any acceptable form
%   - X:        The sample matrix of (source) nodes (2xn or 3xn)
%   - Xt:       The sample matrix of (target) nodes (2xn or 3xn)
%   - ppedges:  The cell array of parameters for plotting graph edges
%   - ppsn:     The cell array of parameters for plotting source nodes
%   - pptn:     The cell array of parameters for plotting target nodes
%   - h:        The vector of handles to all plotted objects
%
% $ Description $
%   - sldrawgraph(G, X) draws a graph with the coordinates of the 
%     graph nodes given in X (a 2 x n matrix for 2D points or 3 x n
%     matrix for 3D points), using default plotting parameters.
%
%   - sldrawgraph(G, X, Xt) draws a bigraph using default plotting
%     parameters.
%
%   - sldrawgraph(G, X, [], ppedges) draws a graph using the specified
%     parameters to draw edges.
%
%   - sldrawgraph(G, X, [], ppedges, ppsn) draws a graph using the 
%     specified parameters to draw edges and nodes.
%
%   - sldrawgraph(G, X, Xt, ppedges) draws a bigraph using the specified
%     parameters to draw edges.
%
%   - sldrawgraph(G, X, Xt, ppedges, ppsn, pptn) draws a bigraph using the 
%     specified parameters to draw edges and nodes.
%
%   - h = sldrawgraph(...) returns the handles to the plotted objects.
%
% $ History $
%   - Created by Dahua Lin, on Sep 11st, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('sldrawgraph', 2);
end

EG = sledgeset(G, 'off');
gi = slgraphinfo(EG);
n = gi.n;
nt = gi.nt;

d = size(X, 1);
if ~isnumeric(X) || ndims(X) ~= 2 || (d ~= 2 && d ~= 3)
    error('sltoolbox:invalidarg', ...
        'The X should be a 2D numeric matrix, and should have 2 or 3 rows.');
end

if nargin < 3 || isempty(Xt)
    Xt = X;
    bi = false;
else
    if ~isnumeric(Xt) || ndims(Xt) ~= 2
        error('sltoolbox:invalidarg', ...
            'The X should be a 2D numeric matrix, and should have 2 or 3 rows.');
    end
    if size(Xt, 1) ~= d
        error('sltoolbox:sizmismatch', ...
            'The sample dimensions in Xt is not the same as X');
    end
    bi = true;
end

if size(X, 2) ~= n || size(Xt, 2) ~= nt
    error('sltoolbox:sizmismatch', ...
        'The size of graph is not consistent with the sample numbers');
end


if nargin < 4 || isempty(ppedges)
    ppedges = {};
end

if nargin < 5 || isempty(ppsn)
    ppsn = {};
end

if nargin < 6 || isempty(pptn)
    pptn = ppsn;
end

if nargout >= 1
    output_h = true;
else
    output_h = false;
end

%% Prepare Data for plotting 

% produce pruned NaN-separated coordinate list

edges = EG.edges;
edges = slpruneedgeset(n, nt, edges);

I = edges(:,1);
J = edges(:,2);
nedges = length(I);

xc = [X(1, I); Xt(1, J); NaN(1, nedges)];
xc = xc(:);
yc = [X(2, I); Xt(2, J); NaN(1, nedges)];
yc = yc(:);
if d == 3
    zc = [X(3, I); Xt(3, J); NaN(1, nedges)];
    zc = zc(:);
end

%% plot edges

% plot the edges
if d == 2
    ch = plot(xc, yc, ppedges{:});
else
    ch = plot3(xc, yc, zc, ppedges{:});
end
if output_h
    h = ch;
end
clear xc yc zc;

%% plot source nodes

if ~isempty(ppsn)
    ppsn = [ppsn, {'LineStyle', 'none'}];
    hold on;
    if d == 2
        ch = plot(X(1,:), X(2,:), ppsn{:});
    else
        ch = plot3(X(1,:), X(2,:), X(3,:), ppsn{:});
    end
    if output_h
        h = [h; ch];
    end
end

%% plot target nodes

if bi && ~isempty(pptn)
    pptn = [pptn, {'LineStyle', 'none'}];
    hold on;
    if d == 2
        ch = plot(Xt(1,:), Xt(2,:), pptn{:});
    else
        ch = plot3(Xt(1,:), Xt(2,:), Xt(3,:), pptn{:});
    end
    if output_h
        h = [h; ch];
    end
end

   
