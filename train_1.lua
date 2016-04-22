require 'torch'
require 'nn'
require 'optim'
require 'sys'

function parse_inputfile(filename) 
	-- Returns the list of all words in an unordered set
	print("===> Parsing input file " .. filename)

	fh, err = io.open(filename)
	if err then print ("Error in opening file " .. filename) return end
	
	maxSeqLen = 0
	lines = 0

	vocab = {}
	while true do
		local line = fh:read()
		if line == nil then break end 
		lines  = lines + 1
		templen = 0
		for word in string.gmatch(line, "%w+") do 
			templen = templen + 1
			if vocab[string.lower(word)] == nil then
				vocab[string.lower(word)] = true
			end
		end
		if templen > maxSeqLen then
			maxSeqLen = templen
		end
	end

	info = {}
	info.vocab = vocab
	info.maxSeqLen = maxSeqLen
	info.inputSize = lines

	return info
end

function parse_targetfile(filename)
	print("===> Parsing target file " .. filename)

	fh, err = io.open(filename)
	if err then print("Error in opening file " .. filename) return end

	labels_table = {}
	labels = 0
	lines = 0

	while true  do
		local line = fh:read()
		if line == nil then break end
		lines = lines + 1

		for word in string.gmatch(line, "%w+") do
			if labels_table[word] == nil then
				assert(tonumber(word), "Invalid input: " .. word .. " is not a number")
				table.insert(labels_table, tonumber(word))
				labels = labels + 1 
			end
		end
	end

	info = {}
	info.labels = labels
	info.labels_table = labels_table 
	info.targetSize = lines

	return info
end

function parse_vocab(vocab)
	-- Converts unsorted set of vocab words to dictionary of word -> ID
	temp = {}
	sorted = {}
	for i, k in pairs(vocab) do
		table.insert(temp, i)
	end
	table.sort(temp)
	for i, k in ipairs(temp) do
		sorted[k] = i
	end
	return sorted
end

function create_model1(maxSeqLen, labels)
	print("===> Creating model1")
	local model1 = nn.Sequential()
	while maxSeqLen > 3*labels do -- TODO replace with number of labels
		model1:add(nn.Linear(maxSeqLen, maxSeqLen/3))
		model1:add(nn.Dropout(0.5))
		maxSeqLen = maxSeqLen/3
	end
	model1:add(nn.Linear(maxSeqLen, 3))
	model1:add(nn.Dropout(0.5))
	model1:add(nn.LogSoftMax())
	model1:remove(2)
	model1:insert(nn.Dropout(0.2), 2)
	print(model1)
	return model1
end

cmd = torch.CmdLine()
cmd:text()
cmd:text("Train a neural network for text classifcation")
cmd:text()
cmd:text("Options:")
cmd:option("--inputfile", "data/all_questions.txt", "filename of inputs")
cmd:option("--targetfile", "data/all_labels.txt", "filename of targets")
cmd:option("--cuda", false, "enable CUDA")
cmd:option("--batchSize", 10, "choose batchsize during training")
cmd:option("--epochs", 10, "choose number of epochs during training")
cmd:text()

opt = cmd:parse(arg) 

if opt.cuda then
	require 'cutorch'
end

inputInfo = parse_inputfile(opt.inputfile)
vocab = parse_vocab(inputInfo.vocab) 
maxSeqLen = inputInfo.maxSeqLen 

targetInfo = parse_targetfile(opt.targetfile) 
labels = targetInfo.labels
labels_table = targetInfo.labels_table

assert(targetInfo.targetSize == inputInfo.inputSize, "The number of inputs (" .. inputInfo.inputSize .. ") and targets (" .. targetInfo.targetSize .. ") must be equal")

model = create_model1(maxSeqLen) 
criterion = nn.ClassNLLCriterion()

if model then
	parameters, gradParameters = model:getParameters()
end

if opt.cuda then
	model = model:cuda()
	criterion = criterion:cuda()
end

trSize = inputInfo.inputSize
adamOptimState = {
	lr = 0.05,
	beta1 = 0.1
}	

shuffle = torch.randperm(trSize)


print("===> Begining training")

for ep = 1,epoch do
	for t=1,trSize,batchSize do
		xlua.progress(t, trsize)

		inputs = {}
		targets = {}

		for i = t,math.min(t+batchSize-1,trSize) do
			-- TODO finish training code!! 
		end

		feval = function(x)
			gradParameters:zero()
			local f = 0
			for i = 1,#inputs do
				local output = model:forward(inputs[i])
				local err = criterion:forward(output, targets[i])
				f = f + err 
				local df_do = criterion:backward(output, targets[i])
				model:backward(inputs[i], df_do)
			end

			gradParameters:div(#inputs)
			f = f/#inputs

			print("\nloss: " .. f)
			return f,gradParameters 
		end 

	end
end
