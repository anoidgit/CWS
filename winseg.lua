print("set default tensor type to float")
torch.setdefaulttensortype('torch.FloatTensor')

function gradUpdate(mlpin, x, y, criterionin, learningRate)
	local pred=mlpin:forward(x)
	local err=criterionin:forward(pred, y)
	sumErr=sumErr+err
	local gradCriterion=criterionin:backward(pred, y)
	mlpin:zeroGradParameters()
	mlpin:backward(x, gradCriterion)
	mlpin:updateGradParameters(0.875)
	mlpin:updateParameters(learningRate)
	mlpin:maxParamNorm(-1)
end

function graphmodule(module_graph)
	local input=nn.Identity()()
	local output=module_graph(input)
	return nn.gModule({input},{output})
end

function loadseq(fname)
	local file=io.open(fname)
	local num=file:read("*n")
	local rs={}
	while num do
		local tmpt={}
		for i=1,num do		
			local vi=file:read("*n")
			table.insert(tmpt,vi)
		end
		table.insert(rs,tmpt)
		num=file:read("*n")
	end
	file:close()
	return rs
end

function easytrainseq()
	local azadd=math.floor(winsize/2)
	local szadd=azadd-1
	local rst={}
	local i=1
	for i=1,nsam do
		local seqex={}
		local j=1
		for j=1,szadd do
			table.insert(seqex,nvec)
		end
		for i,v in ipairs(trainseq[i]) do
			table.insert(seqex,v)
		end
		for j=1,azadd do
			table.insert(seqex,nvec)
		end
		table.insert(rst,seqex)
	end
	trainseq=rst
end

function easytarseq()
	local rst={}
	local i=1
	for i=1,#tarseq do
		table.remove(tarseq[i],1)
		table.insert(rst,torch.Tensor(tarseq[i]))
	end
	tarseq=rst
end

function fillsamplecache()
	while #samicache<cachesize do
		local sid=torch.random(1,nsam)
		local seqex=trainseq[sid]
		for i=0,#seqex-winsize do
			local tmptable={}
			local j=1
			for j=1,winsize do
				table.insert(tmptable,seqex[i+j])
			end
			table.insert(samicache,torch.Tensor(tmptable))
		end
		local ttar=tarseq[sid]
		for i=1,(#ttar)[1] do
			table.insert(samtcache,ttar[i])
		end
	end
end

function getsamples()
	if #samicache<batchsize then
		fillsamplecache()
	end
	local inp={}
	local tar={}
	for i=1,batchsize do
		local inpu={}
		local tmpi=table.remove(samicache,1)
		for j=1,winsize do
			table.insert(inpu,tmpi[j])
		end
		table.insert(inp,inpu)
		table.insert(tar,table.remove(samtcache,1))
	end
	return torch.Tensor(inp),torch.Tensor(tar):resize(batchsize,1)
end

function loadObject(fname)
	local file=torch.DiskFile(fname)
	local objRd=file:readObject()
	file:close()
	return objRd
end

function saveObject(fname,objWrt)
	local file=torch.DiskFile(fname,'w')
	objWrt:lightSerial()
	file:writeObject(objWrt)
	file:close()
end

print("load settings")
winsize=7
batchsize=1024
modlr=0.5

print("load training data")
trainseq=loadseq('datasrc/luamsrtrain.txt')
tarseq=loadseq('datasrc/luamsrtarget.txt')

print("load vectors")
wvec=loadObject('datasrc/wvec.asc')

cachesize=batchsize*4

nvec=(#wvec)[1]
sizvec=(#wvec)[2]
samicache={}
samtcache={}
nsam=#trainseq

sumErr=0
crithis={}
erate=0
storemini=1
storenleg=1
ieps=256
totrain=ieps*batchsize
minerrate=0.00035

print("prefit train data")
easytrainseq()
easytarseq()

print("load packages")
require "nn"
require "nngraph"
require "dpnn"
require "vecLookup"
require "gnuplot"

print("design neural networks")
function getnn()
	local isize=sizvec*winsize
	local hsize=math.floor(isize*0.5)
	--local hfsize=math.floor(hsize*0.5)
	local nnmodinput=nn.Sequential()
		:add(nn.vecLookup(wvec))
		:add(nn.Reshape(isize,true))
	local nnmodcore=graphmodule(nn.Sequential()
		:add(nn.Linear(isize,hsize))
		:add(nn.Tanh())
	--	:add(nn.Linear(hsize,hfsize))
	--	:add(nn.Tanh())
	--	:add(nn.Linear(hfsize,1))
		:add(nn.Linear(hsize,1))
		:add(nn.Sigmoid()))

	local nnmod=nn.Sequential()
		:add(nnmodinput)
		:add(nnmodcore)
	return nnmod
end

nnmod=getnn()

print(nnmod)

print("design criterion")
critmod = nn.BCECriterion()

print("start train")
epochs=1

print("start pre train")
lr=modlr
for tmpi=1,32 do
	for tmpi=1,ieps do
		input,target=getsamples(batchsize)
		gradUpdate(nnmod,input,target,critmod,lr)
	end
	erate=sumErr/totrain
	print("epoch:"..tostring(epochs)..",lr:"..lr..",PPL:"..erate)
	table.insert(crithis,erate)
	sumErr=0
	epochs=epochs+1
end

epochs=1
icycle=1

aminerr=0
lrdecayepochs=1

while true do
	print("start innercycle:"..icycle)
	for innercycle=1,256 do
		for tmpi=1,ieps do
		input,target=getsamples(batchsize)
		gradUpdate(nnmod,input,target,critmod,lr)
		end
		erate=sumErr/totrain
		print("epoch:"..tostring(epochs)..",lr:"..lr..",PPL:"..erate)
		table.insert(crithis,erate)
		if erate<minerrate and erate~=0 then
			print("new minimal error found,save model")
			minerrate=erate
			saveObject("winrs/nnmod"..storemini..".asc",nnmod)
			storemini=storemini+1
			aminerr=0
		else
			if aminerr>=4 then
				aminerr=0
				lrdecayepochs=lrdecayepochs+1
				lr=modlr/(lrdecayepochs)
			end
			aminerr=aminerr+1
		end
		sumErr=0
		epochs=epochs+1
	end

	print("save neural network trained")
	saveObject("winrs/nnmod.asc",nnmod)

	print("save criterion history trained")
	critensor=torch.Tensor(crithis)
	saveObject("winrs/crit.asc",critensor)

	print("plot and save criterion")
	gnuplot.plot(critensor)
	gnuplot.figprint("winrs/crit.png")
	gnuplot.figprint("winrs/crit.eps")
	gnuplot.plotflush()

	print("task finished!Minimal error rate:"..minerrate)

	print("collect garbage")
	collectgarbage()

	print("wait for test, neural network saved at nnmod*.asc")

	icycle=icycle+1

end
