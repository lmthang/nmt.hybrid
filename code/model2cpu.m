function model2cpu(inFile, outFile)

load(inFile);
fields = fieldnames(model);
for ii=1:length(fields)
  field = fields{ii}
  if iscell(model.(field))
    for jj=1:length(model.(field))
      model.(field){jj}= gather(model.(field){jj});
    end
  else
    model.(field) = gather(model.(field));
  end
end
save(outFile, 'model', 'params');

