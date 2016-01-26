function [params] = prepareVocabs(params)
% prepareVocabs - prepare vocabs for the model
%
% Input:
%   params: parameter settings
%
% Output:
%   params: updated parameter settings with vocab fields
%
% Authors: 
%   Thang Luong @ 2015, <lmthang@stanford.edu>
%

  %% grad check
  if params.isGradCheck
    if params.isBi
      srcVocab = {'<s>', '</s>', 'x', 'y'};
      params.srcSos = 1;
    end
    tgtVocab = {'<s>', '</s>', 'a', 'b'};
    params.tgtSos = 1;
    params.tgtEos = 2;
  else
    if params.isBi
      [srcVocab] = loadVocab(params.srcVocabFile);
      srcVocab{end+1} = '<s_sos>'; % not learn
      params.srcSos = length(srcVocab);
    end
    
    [tgtVocab] = loadVocab(params.tgtVocabFile);  
    tgtVocab{end+1} = '<t_sos>';
    params.tgtSos = length(tgtVocab);
    tgtVocab{end+1} = '<t_eos>';
    params.tgtEos = length(tgtVocab); 
  end
  
  %% finalize vocab
  if params.isBi
    params.srcVocab = srcVocab;
    params.srcVocabSize = length(srcVocab);
  end
  
  params.tgtVocab = tgtVocab;
  params.tgtVocabSize = length(tgtVocab);
end