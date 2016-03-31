function FB = slgaborbands(w, scales, orientations)
%SLGABORBANDS Generates a set of Gabor kernels
%
% $ Syntax $
%   - FB = slgaborbands(w, scales, radians)
%
% $ Arguments $
%   - w:                   the kernel window size is wxw
%   - scales:              the scales (number of scales is m)
%   - orientations:        the orientations (number of orientations is o)
%   - FB:                  the array of filter banks
%
% $ Description $
%   - Generates a set of complex filter banks for Gabor, with the 
%     scales and orientations of the filter specified. If there are
%     m scales and n orientations, it will product an w x w x m x n
%     array containing m x n kernels.
%
% $ References $
%   - C. Liu and H. Wechsler, ''Gabor Feature Based Classification Using the
%   Enhanced Fisher Linear Discriminant Model for Face Recognition'', IEEE
%   Trans. on Imag. Proc, Vol. 11, No. 4, Apr, 2002.
%
% $ History $
%   - Created by Dahua Lin on Feb 25th, 2006.
%   - Modified by Dahua Lin on Aug 4th, 2006.
%


m = length(scales);
n = length(orientations);
FB = zeros(w, w, m, n);

for i = 1 : m
    for j = 1 : n
        FB(:,:,i,j) = single_gabor_kernel(w, scales(i), orientations(j));
    end
end

function M = single_gabor_kernel(w, v, u)
% v - scale, u - orientation

mincoord = fix(- w / 2);
coords = mincoord + (0 : w-1);
[X, Y] = meshgrid(coords, coords);

kmax = pi / 2;
f = sqrt(2);
sigma = 2 * pi;

k_v = kmax / f^v;
phi_u = pi * u / 8; 
k_uv = k_v * exp(i * phi_u);

F1 = (k_v^2) / (sigma^2) * exp(-k_v^2 * abs(X.^2 + Y.^2) / (2*sigma^2)) ;
F2 = exp(i * (real(k_uv) * X + imag(k_uv) * Y)) - exp(-sigma^2/2);
M = F1 .* F2;






