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

function evaDev(mlpin, x, y, criterionin)
	mlpin:evaluate()
	return criterionin:forward(mlpin:forward(x), y)
end

function getresmodel(modelcap,scale,usegraph)
	local rtm=nn.ConcatTable()
	rtm:add(modelcap)
	if not scale or scale==1 then
		rtm:add(nn.Identity())
	elseif type(scale)=='number' then
		rtm:add(nn.Sequential():add(nn.Identity()):add(nn.MulConstant(scale,true)))
	else
		rtm:add(nn.Sequential():add(nn.Identity()):add(scale))
	end
	local rsmod=nn.Sequential():add(rtm):add(nn.CAddTable())
	if usegraph then
		local input=nn.Identity()()
		local output=rsmod(input)
		return nn.gModule({input},{output})
	else
		return rsmod
	end
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

function easyinputseq(seqin)
	local azadd=math.floor(winsize/2)
	local szadd=azadd-1
	local rst={}
	local i=1
	for i=1,#seqin do
		local seqex={}
		local j=1
		for j=1,szadd do
			table.insert(seqex,nvec)
		end
		for _,v in ipairs(seqin[i]) do
			table.insert(seqex,v)
		end
		for j=1,azadd do
			table.insert(seqex,nvec)
		end
		table.insert(rst,seqex)
	end
	return rst
end

function easytarseq(seqin)
	local rst={}
	local i=1
	for i=1,#seqin do
		table.remove(seqin[i],1)
		table.insert(rst,seqin[i])
	end
	return rst
end

function fillsamplecache()
	while #samicache<cachesize do
		local sid=torch.random(1,nsam)
		local seqex=trainseq[sid]
		for i=0,#seqex-winsize do
			table.insert(samicache,{table.unpack(seqex,i+1,i+winsize)})
		end
		for _,v in ipairs(tarseq[sid]) do
			table.insert(samtcache,v)
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
		table.insert(inp,table.remove(samicache,1))
		table.insert(tar,table.remove(samtcache,1))
	end
	return torch.Tensor(inp),torch.Tensor(tar):resize(batchsize,1)
end

function loadDev(inpf,tarf)
	local devseq=easyinputseq(loadseq(inpf))
	local devtar=easytarseq(loadseq(tarf))
	local devinp={}
	local devt={}
	local seqid=1
	for seqid=1,#devseq do
		local seqex=devseq[seqid]
		for i=0,#seqex-winsize do
			table.insert(devinp,{table.unpack(seqex,i+1,i+winsize)})
		end
		for _,v in ipairs(devtar[seqid]) do
			table.insert(devt,v)
		end
	end
	return torch.Tensor(devinp),torch.Tensor(devt):resize(#devt,1)
end

function loadObject(fname)
	local file=torch.DiskFile(fname)
	local objRd=file:readObject()
	file:close()
	return objRd
end

function saveObject(fname,objWrt)
	if not objWrt:isTensor() then
		objWrt:lightSerial()
	end
	local file=torch.DiskFile(fname,'w')
	file:writeObject(objWrt)
	file:close()
end

print("load settings")
winsize=7
batchsize=1024
ieps=256
modlr=0.5

print("load vectors")
wvec=loadObject('datasrc/wvec.asc')

nvec=(#wvec)[1]
sizvec=(#wvec)[2]

print("load training data")
trainseq=easyinputseq(loadseq('datasrc/luatrain.txt'))
tarseq=easytarseq(loadseq('datasrc/luatarget.txt'))

devin,devt=loadDev('datasrc/luadevtrain.txt','datasrc/luadevtarget.txt')

nsam=#trainseq

cachesize=batchsize*4

samicache={}
samtcache={}

sumErr=0
crithis={}
cridev={}
erate=0
edevrate=0
storemini=1
storedevmini=1
minerrate=0.875
mindeverrate=0.875

print("load packages")
require "nn"
require "nngraph"
require "dpnn"
require "vecLookup"
require "gnuplot"

print("design neural networks")
function getnn()
	local picwidth=7
	local picheight=7
	local picdepth=4
	local nifilter=16

	local nifilter2=nifilter*2
	local nifilter4=nifilter2*2
	local nifilter8=nifilter4*2

	local isize=sizvec*winsize
	local picsize=picdepth*picheight*picwidth
	local mtsize=math.floor((isize+picsize)/2)
	local cosize=nifilter2*(picheight-2-2-2)*(picwidth-2-2-2)

	local srtanh=getresmodel(nn.Tanh(),0.125,true)

	local nnmodinput=nn.Sequential()
		:add(nn.vecLookup(wvec))
		:add(nn.Reshape(isize,true))

	local nnmodcore=nn.Sequential()
		:add(nn.Linear(isize,mtsize))
		:add(srtanh:clone())
		:add(nn.Linear(mtsize,picsize))
		:add(srtanh:clone())
		:add(nn.Reshape(picdepth,picheight,picwidth,true))
		:add(nn.SpatialConvolution(picdepth, nifilter, 3, 1))
		:add(srtanh:clone())
		:add(nn.SpatialConvolution(nifilter, nifilter2, 1, 3))
		:add(srtanh:clone())
		:add(nn.SpatialConvolution(nifilter2, nifilter, 1, 1))
		:add(srtanh:clone())
		:add(nn.SpatialConvolution(nifilter, nifilter2, 3, 1))
		:add(srtanh:clone())
		:add(nn.SpatialConvolution(nifilter2, nifilter4, 1, 3))
		:add(srtanh:clone())
		:add(nn.SpatialConvolution(nifilter4, nifilter2, 1, 1))
		:add(srtanh:clone())
		:add(nn.SpatialConvolution(nifilter2, nifilter4, 3, 1))
		:add(srtanh:clone())
		:add(nn.SpatialConvolution(nifilter4, nifilter2, 1, 1))
		:add(srtanh:clone())
		:add(nn.SpatialConvolution(nifilter2, nifilter4, 1, 3))
		:add(srtanh:clone())
		:add(nn.SpatialConvolution(nifilter4, nifilter2, 1, 1))
		:add(nn.Convert('bchw','bf'))
		:add(nn.Tanh())
		:add(nn.Linear(cosize,1))
		--:add(nn.Sigmoid())

	local nnmod=nn.Sequential()
		:add(nnmodinput)
		:add(graphmodule(nnmodcore))

	return nnmod
end

nnmod=getnn()

print(nnmod)

print("design criterion")
critmod = nn.MarginCriterion()
--if use the BCECriterion below, uncomment the Sigmoid activation funcition in getnn()
--critmod = nn.BCECriterion()

print("init train")
epochs=1
lr=modlr
collectgarbage()

print("start pre train")
for tmpi=1,32 do
	nnmod:training()
	for tmpi=1,ieps do
		input,target=getsamples(batchsize)
		gradUpdate(nnmod,input,target,critmod,lr)
	end
	erate=sumErr/ieps
	table.insert(crithis,erate)
	edevrate=evaDev(nnmod,devin,devt,critmod)
	table.insert(cridev,edevrate)
	print("epoch:"..tostring(epochs)..",lr:"..lr..",PPL:"..erate..",Dev:"..edevrate)
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
		nnmod:training()
		for tmpi=1,ieps do
			input,target=getsamples(batchsize)
			gradUpdate(nnmod,input,target,critmod,lr)
		end
		erate=sumErr/ieps
		table.insert(crithis,erate)
		edevrate=evaDev(nnmod,devin,devt,critmod)
		table.insert(cridev,edevrate)
		print("epoch:"..tostring(epochs)..",lr:"..lr..",PPL:"..erate..",Dev:"..edevrate)
		if erate<minerrate then
			print("new minimal error found,save model")
			minerrate=erate
			saveObject("reconvrs/nnmod"..storemini..".asc",nnmod)
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
		if edevrate<mindeverrate then
			print("new minimal dev error found,save model")
			mindeverrate=edevrate
			saveObject("reconvrs/devnnmod"..storedevmini..".asc",nnmod)
			storedevmini=storedevmini+1
		end
		sumErr=0
		epochs=epochs+1
	end

	print("save neural network trained")
	saveObject("reconvrs/nnmod.asc",nnmod)

	print("save criterion history trained")
	critensor=torch.Tensor(crithis)
	saveObject("reconvrs/crit.asc",critensor)
	critdev=torch.Tensor(cridev)
	saveObject("reconvrs/critdev.asc",critdev)

	print("plot and save criterion")
	gnuplot.plot(critensor)
	gnuplot.figprint("reconvrs/crit.png")
	gnuplot.figprint("reconvrs/crit.eps")
	gnuplot.plotflush()
	gnuplot.plot(critdev)
	gnuplot.figprint("reconvrs/critdev.png")
	gnuplot.figprint("reconvrs/critdev.eps")
	gnuplot.plotflush()

	print("task finished!Minimal error rate:"..minerrate)

	print("wait for test, neural network saved at nnmod*.asc")

	icycle=icycle+1

	print("collect garbage")
	collectgarbage()

end
