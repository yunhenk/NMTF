import numpy as np
from sklearn.cluster import KMeans
import pickle
import math

data_path='hashtag/Basing_AR_user_Part/'

def readFromFile(UPath):
    U=np.load(UPath)
    dealWith(U)

def dealWith(U):
    classlabels=np.zeros(U.shape[0])
    for i in range(0,len(U)):
        maxindex=1
        maxvalue=U[i][0]
        for j in range(1,len(U[0])):
            if U[i][j]>maxvalue:
                maxindex=j+1
                maxvalue=U[i][j]
            if U[i][j]<0:
                U[i][j]=0
        classlabels[i]=maxindex
    #print(U[0:5])s
    #print(classlabels)
    #purity=calcuPurity(classlabels)
    density=calcuDensity(len(U[0]),classlabels)
    return density

def dealWithEmbedding(W_h):
    n_clusters=8
    random_state=170
    y_pred=KMeans(n_clusters=n_clusters,random_state=random_state).fit_predict(W_h)
    #print(y_pred)sf
    #print(len(y_pred))

    temp=np.add(y_pred,1)
    #print(temp)


    #hscore=calcuHscore(W_h,n_clusters,temp)

    puri=calcuPurity(temp)
    density=calcuDensity(n_clusters,temp)
    #return density
    return puri

def dealWithAllEmbedding(W_h):
    hids=np.load(data_path+'hids.npy')
    cindexs=np.load(data_path+'cindexs.npy')
    embeddings=[]
    for index in cindexs:
        embeddings.append(W_h[index])

    n_clusters=8
    random_state=170
    y_pred=KMeans(n_clusters=n_clusters,random_state=random_state).fit_predict(embeddings)
    temp=np.add(y_pred,1)
    puri=calcuPurity(temp)
    calcuDensitySparse(n_clusters,temp)
    return puri

def dealWithAllEmbeddingFromJava(file_path):
    return 1

def dealWithAllEmbeddingSparse(W_h):
    dimension=W_h.w
    cindexs=np.load(data_path+'cindexs.npy')
    embeddings=np.zeros([len(cindexs),dimension])
    W_h=W_h.getRC1Form()
    for i in range(0,len(cindexs)):
        index=cindexs[i]
        rowcontent=W_h.content[index]
        for c,value in rowcontent.items():
            embeddings[i][c]=value

    n_clusters=8
    random_state=170
    y_pred=KMeans(n_clusters=n_clusters,random_state=random_state).fit_predict(embeddings)
    temp=np.add(y_pred,1)
    puri=calcuPurity(temp)
    calcuDensitySparse(n_clusters,temp)
    return puri

def calcuPurity(classlabels):
    #print data_path
    hlabels=np.load(data_path+'hlabels.npy')
    #print(hlabels)
    cc=getCommunitys(classlabels)
    hc=getCommunitys(hlabels)
    sumi=0
    for k,v in cc.items():
        numofinter=-1
        inter=[]
        inter=set(inter)
        for hk,hv in hc.items():
            inter=v.intersection(hv)
            if len(inter)>numofinter:
                numofinter=len(inter)
        sumi+=numofinter
    purity=float(sumi)/len(hlabels)
    print('Purity is: '+str(purity))
    return purity

def getCommunitys(classlabels):
    comm={}
    for i in range(0,len(classlabels)):
        label=classlabels[i]
        ins=[]
        ins=set(ins)
        if comm.has_key(label):
            ins=comm[label]
            ins.add(i)
            comm[label]=ins
        else:
            ins.add(i)
            comm[label]=ins
    return comm

def calcuDensity(n_clusters,y_pred):

    map={}
    for i in range(0,n_clusters):
        indexs=[]
        map[i]=indexs
    for i in range(0,len(y_pred)):
        clusterId=y_pred[i]
        indexs=map[clusterId-1]
        indexs.append(i)
        map[clusterId-1]=indexs

    print map
    for k,v in map.items():
        print len(v)
    hhmatrix_path=data_path+'M_u_u.npy'
    hhmatrix=np.load(hhmatrix_path)
    count=0
    for i in map:
        indexs=map[i]
        for j in range(0,len(indexs)):
            for k in range(0,len(indexs)):
                index1=indexs[j]
                index2=indexs[k]
                if(hhmatrix[index1][index2]!=0):
                    count=count+1
    countE=0
    for i in range(0,len(hhmatrix)):
        for j in range(0,len(hhmatrix[0])):
            if(hhmatrix[i][j]!=0):
                countE=countE+1
    #print(count)
    #print(countE)
    density=float(count)/countE
    print('Density is: '+str(density))
    return density

def calcuDensitySparse(n_clusters,y_pred):

    map={}
    for i in range(0,n_clusters):
        indexs=[]
        map[i]=indexs
    cindexs=np.load(data_path+'cindexs.npy')
    for i in range(0,len(y_pred)):
        clusterId=y_pred[i]
        indexs=map[clusterId-1]
        indexs.append(cindexs[i])
        map[clusterId-1]=indexs

    hhmatrix_path=data_path+'M_h_h_n_sparse.txt'
    f=file(hhmatrix_path,'rb')
    M_h_h_n_sparse=pickle.load(f)
    count=0
    for i in map:
        indexs=map[i]
        for j in range(0,len(indexs)):
            for k in range(0,len(indexs)):
                index1=indexs[j]
                index2=indexs[k]
                rowcontent=M_h_h_n_sparse.content[index1]
                if rowcontent.has_key(index2):
                    count=count+1
    countE=0
    for index in cindexs:
        rowcontent=M_h_h_n_sparse.content[index]
        for c,value in rowcontent.items():
            if c in cindexs:
                countE=countE+1
    #print(count)
    #print(countE)
    density=float(count)/countE
    print('Density is: '+str(density))

def cos_simi(a,b):
	if(len(a)!=len(b)):
		return None
	part_up=0.0
	a_sq=0.0
	b_sq=0.0
	for i,j in zip(a,b):
		part_up+=i*j
		a_sq+=i*i
		b_sq+=j*j
	part_down=math.sqrt(a_sq*b_sq)
	if(part_down==0.0):
		return None
	else:
		return part_up/part_down

def calcuHscoreBasingLabels(rep,n_clusters):
    hlabels=np.load(data_path+'hlabels.npy')
    return calcuHscore(rep,n_clusters,hlabels)

#
def calcuHscore(rep,n_clusters,y_pred):

    map={}
    for i in range(0,n_clusters):
        indexs=[]
        map[i]=indexs
    for i in range(0,len(y_pred)):
        clusterId=y_pred[i]
        indexs=map[clusterId-1]
        indexs.append(i)
        map[clusterId-1]=indexs

    allinclus=0.0
    for i in map:
        indexs=map[i]
        inclus=0.0
        for j in range(0,len(indexs)):
            for k in range(j+1,len(indexs)):
                index1=indexs[j]
                index2=indexs[k]
                cosjk=1-cos_simi(rep[index1],rep[index2])
                inclus+=cosjk*2/(len(indexs)*(len(indexs)-1))
        allinclus+=inclus
    averinclus=allinclus/len(map)
    #print(averinclus)

    alloutclus=0.0
    for i in range(0,len(map)):
        for j in range(i+1,len(map)):
            indexsi=map[i]
            indexsj=map[j]
            outclus=0.0
            for k in range(0,len(indexsi)):
                for l in range(0,len(indexsj)):
                    index1=indexsi[k]
                    index2=indexsj[l]
                    coskl=1-cos_simi(rep[index1],rep[index2])
                    outclus+=coskl*2/(len(indexsi)*len(indexsj))
            alloutclus+=outclus
    averoutclus=alloutclus/((len(map)*(len(map)-1)))
    #print(averoutclus)
    h_score=averinclus/averoutclus
    print('h_score is:'+str(h_score))
    return h_score

def run():
    readFromFile('hashtag/U/U-2016-10-27-22:50:12.npy')

def showResult():
    U_path='hashtag/U/U-2016-10-31-21:15:43.npy'
    U=np.load(U_path)
    classlabels=np.zeros(U.shape[0])
    for i in range(0,len(U)):
        maxindex=1
        maxvalue=U[i][0]
        for j in range(1,len(U[0])):
            if U[i][j]>maxvalue:
                maxindex=j+1
                maxvalue=U[i][j]
            if U[i][j]<0:
                U[i][j]=0
        classlabels[i]=maxindex
    cc=getCommunitys(classlabels)
    data_path='hashtag/Basing_AR_both_Part/'
    printClusters(cc,data_path)
    hlabels=np.load(data_path+'hlabels.npy')
    hc=getCommunitys(hlabels)
    printClusters(hc,data_path)

def printClusters(cc,data_path):
    print cc
    hids=np.load(data_path+'hids.npy')
    #print hids
    AN_path='hashtag/AN.txt'
    f=open(AN_path)
    hashtagcontent={}
    count=1
    for line in f:
        hashtagcontent[count]=line[1:len(line)-2]
        count+=1
    #print hashtagcontent
    hidscontent={}
    hidsnames=[]
    for id in hids:
        hidscontent[id]=hashtagcontent[id]
        hidsnames.append(hashtagcontent[id])
    print hidscontent
    print hidsnames
    print len(hidsnames)
    for k,v in cc.items():
        print 'class: '+str(k)
        contents=''
        for id in v:
            contents+=hidscontent[hids[id]]+' '
        print contents

def forPatent():
    data_path='hashtag/Basing_AR_both_Part/'
    M_u_u=np.load(data_path+'M_u_u_n.npy')
    print M_u_u[0]
    M_u_f=np.load(data_path+'M_u_f_part.npy')
    print M_u_f[0]



#run()
#showResult()
#forPatent()
#test()