torch.setdefaulttensortype('torch.FloatTensor')

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

function ldtensor(fname)
	local file=torch.DiskFile(fname)
	local rs=file:readObject()
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
			table.insert(seqex,-1)
		end
		for i,v in ipairs(trainseq[i]) do
			table.insert(seqex,v)
		end
		for j=1,azadd do
			table.insert(seqex,-1)
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
	local inpid={}
	local tar={}
	for i=1,batchsize do
		local inpu={}
		local tmpi=table.remove(samicache,1)
		for j=1,winsize do
			local ti=wvec[tmpi[j]]
			for z=1,sizvec do
				table.insert(inpu,ti[z])
			end
		end
		table.insert(inp,inpu)
		table.insert(inpid,tmpi)
		table.insert(tar,table.remove(samtcache,1))
	end
	return torch.Tensor(inp),torch.Tensor(tar):resize(batchsize,1),inpid
end

print("load settings")
winsize=5
batchsize=512

cachesize=batchsize

nvec=(#wvec)[1]
sizvec=(#wvec)[2]
samicache={}
samtcache={}
nsam=#trainseq

print("load training data")
trainseq=loadseq('datasrc/luamsrtrain.txt')
tarseq=loadseq('datasrc/luamsrtarget.txt')

print("load vectors")
wvec=ldtensor('datasrc/wvec.asc')

print("prefit train data")
easytrainseq()
easytarseq()

print("start train")
input,target,wid=getsamples()
--gradUpdate()
--updvec()
