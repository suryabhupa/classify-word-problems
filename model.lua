require 'rnn'
require 'nn'
require 'dp'

version = 10

cmd = torch.CmdLine()
cmd:text()
cmd:text('Classify algebraic word problems by number of unknowns')
cmd:text('Example:')
cmd:text('$> th classify.lua > results.txt')
cmd:text('Options:')
cmd:text('--batchSize', 8, 'numbers of examples per batch')
cmd:text('--rho', 10, 'length of sequence')
cmd:text('--hiddenSize', 100, 'size of hidden state representation')
cmd:text('--nIndex', 100, 'size of vocabulary')
cmd:text('--nClass', 10, 'number of output classes')
cmd:text('--lr', 0.1, 'hyperparameter @TODO what does this do')
cmd:text('--silent', false, 'do not print to stdout')
cmd:text()

local opt = cmd:parse(arg or {})
if not opt.silent then
	table.print(opt)
end

-- simple recurrent neural network
r = nn.Recurrent(
			opt.hiddenSize, nn.Identity(), 
			nn.Linear(hiddenSize, hiddenSize),
			nn.Sigmoid(),
			opt.rho
)

rnn = nn.Sequential()
	:add(nn.LookupTable(opt.nIndex, opt.hiddenSize))
	:add(nn.SplitTable(1, 2))
	:add(nn.Sequencer(r))
	:add(nn.SelectTable(-1)) -- this selects the last time-step of the rnn output structure
	:add(nn.Linear(hiddenSize, nClass))
	:add(nn.LogSoftMax)

-- criterion

criterion = nn.ClassNLLCriterion()

ds = {}
ds.size = 3915 -- TODO what is this
ds.input = torch.LongTensor(ds.size, opt.rho)
ds.target = torch.LongTensor(ds.size)
