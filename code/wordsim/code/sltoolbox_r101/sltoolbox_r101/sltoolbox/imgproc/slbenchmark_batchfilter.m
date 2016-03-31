function recs = slbenchmark_batchfilter(imgsiz, nimgs, filtersiz, nfilters)
%SLBENCHMARK_BATCHFILTER Compares the efficiency of batch filter
%
% Input:
%   imgsiz:     the size of each image
%   nimgs:      the number of images
%   filtersiz:  The size of each filter
%   nfilters:   the list of numbers of filters
%
% History
%   - Created by Dahua Lin, on Sep 2nd, 2006


imgs = rand([imgsiz, nimgs]);
fb = rand([filtersiz, max(nfilters)]);

names = {'imfilter', 'slapplyfilerband'};
methods = {@test_imfilter, @test_slband};
nmethods = length(names);

recs = zeros(length(nfilters), nmethods);

for k = 1 : nmethods
    
    curname = names{k};
    curmethod = methods{k};
    
    disp(['Test ', curname]);
    
    for i = 1 : length(nfilters)
        nf = nfilters(i);
        tic;
        curmethod(imgs, fb(:,:,1:nf));        
        recs(i, k) = toc;        
    end        
end

recs = recs / nimgs;


function test_imfilter(imgs, fb)

nf = size(fb ,3);
R = zeros([size(imgs), nf]);
for i = 1 : nf
    R(:,:,:,i) = imfilter(imgs, fb(:,:,i), 'replicate');
end
clear R;

function test_slband(imgs, fb)

fh = size(fb, 1);
fw = size(fb, 2);
R = slapplyfilterband(imgs, fb, [fh, fw]);
clear R;











