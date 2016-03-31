function [GC, spectrum, CTS] = slltsa(X, G, dl, dg)
%SLLTSA Performs Local Tangent Space Alignment Learning
%
% $ Syntax $
%   - [GC, spectrum] = slltsa(X, G, dl)
%   - [GC, spectrum] = slltsa(X, G, dl, dg)
%   - [GC, spectrum, CTS] = slltsa(...)
%
% $ Arguments $
%   - X:        The input sample matrix
%   - G:        The neighborhood graph
%   - dl:       The local dimension
%   - dg:       The global embedding dimension
%   - GC:       The global embedding coordinates
%   - spectrum: The eigenvalue spectrum of the embedding space
%   - CTS:      The struct of the coordinate transform system
%               it has the following fields:
%                   - means:   d0 x n
%                   - bases:   d0 x dl x n
%                   - ftrans:  dl x dg x n
%                      the forward transforms: local coord -> global coord
%
% $ Description $
%   - [GC, spectrum] = slltsa(X, G, dl) performs the local tangent space
%     alignment learning based on the samples in X and the neighborhood
%     graph in G. By default the global embedding dimension is set to the
%     same as the local dimension.
%
%   - [GC, spectrum] = slltsa(X, G, dl, dg) performs the LTSA with the 
%     local and global embeding dimension respectively given.
%
%   - [GC, spectrum, CTS] = slltsa(...) additionally returns the coordinate
%     transform system.
%
% $ Remarks $
%   - The implementation wraps the three major components:
%       - sllocaltanspace:   learns the local tangent spaces
%       - sllocaltancoords:  produces the local coordinates
%       - sllocalcoordalign: pursues the global embedding by aligning the
%                            local coordinates
%
% $ History $
%   - Created by Dahua Lin, on Sep 13rd, 2006
%

%% parse and verify input arguments

if ~isnumeric(X) || ndims(X) ~= 2
    error('X should be a 2D numeric matrix');
end
n = size(X, 2);

gi = slgraphinfo(G, {[n, n]});

if nargin < 4
    dg = dl;
end

if ~strcmp(gi.form, 'adjmat')
    G = sladjmat(G, 'sparse', true);
end

out_cts = (nargout >= 3);


%% main skeleton

% learn local tangent spaces
[LM, LP] = sllocaltanspace(X, G, dl);

% compute local coordinates
[Y, GM] = sllocaltancoords(LM, LP, X, G, 'T');
if ~out_cts
    clear LM LP;
end

% align local systems to global embedding
if out_cts
    [GC, spectrum, LT] = sllocalcoordalign(GM, Y, dg);
else
    [GC, spectrum] = sllocalcoordalign(GM, Y, dg);
end

% organize output
if out_cts
    CTS.means = LM;
    CTS.bases = LP;
    CTS.ftrans = LT;
end

