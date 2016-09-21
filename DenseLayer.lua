require 'nn'
require 'cudnn'
require 'cunn'
local nninit = require 'nninit'

local DenseLayer, parent = torch.class('nn.DenseLayer', 'nn.Container')

function DenseLayer:__init(nInputs, nChannels, growthRate, stride)
  parent.__init(self)
  self.train = true
  self.nInputs = nInputs
  self.nChannels = nChannels
  self.growthRate = growthRate or nChannels
  stride = stride or 1

  self.net = nn.Sequential()
  self.net:add(cudnn.SpatialBatchNormalization(nChannels))
  self.net:add(cudnn.ReLU(true))
  self.net:add(cudnn.SpatialConvolution(nChannels, self.growthRate, 3, 3, stride, stride, 1, 1))

  self.gradInput = {}
  self.output = {torch.CudaTensor()}
  for i = 1, nInputs  do
      self.output[i+1] = torch.CudaTensor()
  end

  self.modules = {self.net}
end

function DenseLayer:updateOutput(input)
  if type(input) == 'table' then
    -- copy input to a contiguous tensor
    local sz = #input[1]
    sz[2] = self.nChannels
    local input_c = self.net:get(1).gradInput -- reuse the memory to save tmp input
    input_c:resize(sz)
    local nC = 1
    for i = 1, self.nInputs do
      self.output[i] = input[i]
      input_c:narrow(2, nC, input[i]:size(2)):copy(input[i])
      nC = nC + input[i]:size(2)
    end
    -- compute output
    sz[2] = self.growthRate
    self.output[self.nInputs+1]:resize(sz):copy(self.net:forward(input_c))
  else
    local sz = input:size()
    sz[2] = self.growthRate 
    self.output[1]:resizeAs(input):copy(input)
    self.output[2]:resize(sz):copy(self.net:forward(input))
  end

  return self.output
end

function DenseLayer:updateGradInput(input, gradOutput)
  if type(input) == 'table' then
    for i = 1, self.nInputs do
      self.gradInput[i] = gradOutput[i]
    end
    local gOut_net = gradOutput[#gradOutput]
    local input_c = self.net:get(1).gradInput -- the contiguous input is stored in the gradInput during the forward pass
    local gIn = self.net:updateGradInput(input_c, gOut_net)
    local nC = 1
    for i = 1, self.nInputs do
      self.gradInput[i]:add(gIn:narrow(2,nC,input[i]:size(2)))
      nC = nC + input[i]:size(2)
    end
  else
    self.gradInput = gradOutput[1]
    self.gradInput:add(self.net:updateGradInput(input, gradOutput[2]))
  end
  return self.gradInput
end

function DenseLayer:accGradParameters(input, gradOutput, scale)
 scale = scale or 1
 local gOut_net = gradOutput[#gradOutput]
 if type(input) == 'table' then
    -- copy input to a contiguous tensor
   local sz = #input[1]
    sz[2] = self.nChannels
    local input_c = self.net:get(1).gradInput -- reuse the memory to save tmp input
    input_c:resize(sz)
    local nC = 1
    for i = 1, self.nInputs do
      input_c:narrow(2, nC, input[i]:size(2)):copy(input[i])
      nC = nC + input[i]:size(2)
    end
    self.net:accGradParameters(input_c, gOut_net, scale)
  else
    self.net:accGradParameters(input, gOut_net, scale)
  end
end