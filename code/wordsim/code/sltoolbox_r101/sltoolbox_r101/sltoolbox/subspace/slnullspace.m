function [V, Vc] = slnullspace(tar, varargin)
%SLRANGESPACE Determines the null-space of the range of X
%
% $ Syntax $
%   - V = slnullspace(tar)
%   - V = slnullspace(tar, ...)
%   - [V, Vc] = slnullspace(...)
%
% $ Arguments $
%   - tar:      the target, can be a sample matrix or covariance
%   - V:        the basis for the null space
%   - Vc:       the basis for the orthogonal complement of the null space.
%
% $ Description $
%   - V = slnullspace(tar) determines the null space of tar in default 
%     settings. Note that tar can have two forms: a sample matrix or a 
%     covariance given by the syntax {'cov', C}.
%
%   - V = slrangespace(tar, ...) determines the range space of X using
%     the specified dimension determination schemes. The arguments 
%     input following X will be delivered to sldim_by_eigval for dimension
%     determination of principal space (the orthogonal complement to V)
%
%   - [V, Vc] = slrangespace(...) also returns orthogonal complement of V.
%
% $ History $
%   - Created by Dahua Lin on Apr 25, 2005
%

[Vc, V] = slrangespace(tar, varargin{:});
