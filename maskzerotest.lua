require 'rnn'
require 'optim'

inSize = 20
batchSize = 2
hiddenSize = 10
seqLengthMax = 11
numTargetClasses=5
numSeq = 30

x, y1 = {}, {}

for i = 1, numSeq do
   local seqLength = torch.random(1,seqLengthMax)
   local temp = torch.zeros(seqLengthMax, inSize)
   local targets ={}
   if seqLength == seqLengthMax then
         targets = (torch.rand(seqLength)*numTargetClasses):ceil()
   else
      targets = torch.cat(torch.zeros(seqLengthMax-seqLength),(torch.rand(seqLength)*numTargetClasses):ceil())
   end
      temp[{{seqLengthMax-seqLength+1,seqLengthMax}}] = torch.randn(seqLength,inSize)
   table.insert(x, temp)
   table.insert(y1, targets)
end

model = nn.Sequencer(
   nn.Sequential()
      :add(nn.MaskZero(nn.FastLSTM(inSize,hiddenSize),1))
      :add(nn.MaskZero(nn.Linear(hiddenSize, numTargetClasses),1))
      :add(nn.MaskZero(nn.LogSoftMax(),1))
)

criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1))

output = model:forward(x)
print(output[1])

err = criterion:forward(output, y1)
print(err)
