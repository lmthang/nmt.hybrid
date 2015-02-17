function model2cpu(inFile, outFile)

load(inFile);
model.W_emb = gather(model.W_emb);
model.W_soft = gather(model.W_soft);
for ii=1:params.numLayers
  model.W_src{ii} = gather(model.W_src{ii});
  model.W_tgt{ii} = gather(model.W_tgt{ii});
end
save(outFile, 'model', 'params');

