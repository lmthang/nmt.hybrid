function labels = slclassify_eucnn(centers, samples)
%SLCLASSIFY_EUCNN Classifies samples using Euclidena-based NN
%
% $ Syntax $
%   - labels = slclassify_eucnn(centers, samples)
%
% $ Arguments $
%   - centers:          the class centers
%   - samples:          the samples to be classified
%   - labels:           the classified result
%
% $ Description $
%   - labels = slclassify_eucnn(centers, samples) classifies the samples to
%     nearest centers based on Euclidean distances. The output labels
%     indicate which nearest center the samples are classified to.
%
% $ History $
%   - Created by Dahua Lin, on Aug 21, 2006
%

dists = slmetric_pw(centers, samples, 'eucdist');
labels = slclassify(dists, 1:size(dists,1), 'low');
