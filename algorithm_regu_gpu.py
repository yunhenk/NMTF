import numpy as np
import scipy as sp
import sys
import pdb
import numpy as np
import cudamat as cm
import time
import performance as per

def optimization(M_u_u,M_u_f,M_t_f,L_u,L_t,S_u_u,S_u_u_D,S_t_t,S_t_t_D,alpha,beta,k,loss,num_step):
    m=M_u_u.shape[0]
    w=M_u_f.shape[1]
    n=M_t_f.shape[0]
    #random samples from a uniform distribution over [0,1)
    U=np.random.rand(m,k)
    # int8() would reduce the precision of float number, don't do it
    # U=np.int8(U)
    V = np.random.rand(n, k)
    W = np.random.rand(w, k)
    H1 = np.random.rand(k, k)
    H2= np.random.rand(k, k)
    H3= np.random.rand(k, k)

    M_u_u=cm.CUDAMatrix(M_u_u)
    #print(M_t_f)
    #print(np.sum(np.sum(M_t_f)))
    M_u_f=cm.CUDAMatrix(M_u_f)
    M_t_f=cm.CUDAMatrix(M_t_f)
    U=cm.CUDAMatrix(U)
    V=cm.CUDAMatrix(V)
    W=cm.CUDAMatrix(W)
    H1=cm.CUDAMatrix(H1)
    H2=cm.CUDAMatrix(H2)
    H3=cm.CUDAMatrix(H3)

    L_u=cm.CUDAMatrix(L_u)
    L_t=cm.CUDAMatrix(L_t)
    S_u_u=cm.CUDAMatrix(S_u_u)
    S_u_u_D=cm.CUDAMatrix(S_u_u_D)
    S_t_t=cm.CUDAMatrix(S_t_t)
    S_t_t_D=cm.CUDAMatrix(S_t_t_D)


    pvalue=0.00000000000001
    step=0
    maxU=U.asarray()
    maxPurity=per.dealWith(maxU)
    while step<num_step:

        # M_t_f is ok now (didn't change along the process )
        # M_t_f_n=M_t_f.asarray()
        # print(np.sum(np.sum(M_t_f_n)))

        # print('M_u_u')
        # print(M_u_u.asarray())
        # print('M_u_f')
        # print(M_u_f.asarray())
        # print('U')
        #print(U.asarray())
        # print('V')
        # print(V.asarray())
        # print('W')
        # print(W.asarray())
        # print('H1')
        # print(H1.asarray())
        # print('H2')
        # print(H2.asarray())
        # print('H3')
        # print(H3.asarray())
        t=targetFunction(M_u_u,M_u_f,M_t_f,L_u,L_t,U,V,W,H1,H2,H3,alpha,beta).asarray()
        print('loss: '+str(t[0][0]))
        if t<=loss:
            break

        # print(S_u_u.asarray())
        # print(cm.dot(S_u_u,U).asarray())
        #print(L_u.asarray())
        #print(manyDot([U.transpose(),L_u,U]).asarray())
        # print(cm.dot(S_u_u_D,U).asarray())

        #update U
        up=manyDot([M_u_u,U,H1.transpose()]).add(manyDot([M_u_f,W,H3.transpose()])).add(cm.dot(S_u_u,U).mult(alpha))
        psaiU= manyDot([U.transpose(), M_u_u, U, H1.transpose()]).add(manyDot([U.transpose(), M_u_f, W, H3.transpose()])).subtract(manyDot([H1, U.transpose(), U, H1.transpose()])).subtract(manyDot([H3, W.transpose(), W, H3.transpose()])).subtract(manyDot([U.transpose(),L_u,U]).mult(alpha))
        down= manyDot([U,H1,U.transpose(),U,H1.transpose()]).add(manyDot([U,H3,W.transpose(),W,H3.transpose()])).add(cm.dot(S_u_u_D,U).mult(alpha)).add(cm.dot(U, psaiU))
        #make zero plus something for divide
        size=down.shape
        plus=np.ones(size)*pvalue
        plus=cm.CUDAMatrix(plus)
        down.add(plus)
        #both multiply divide and sqrt are elment-wise
        up.divide(down)
        up_cpu=up.asarray()
        up.free_device_memory()
        for i in range(0,len(up_cpu)):
            for j in range(0,len(up_cpu[i])):
                if up_cpu[i][j]<0:
                    up_cpu[i][j]=0
        up=cm.CUDAMatrix(up_cpu)
        U.mult(cm.sqrt(up))

        up.free_device_memory()
        psaiU.free_device_memory()
        down.free_device_memory()
        plus.free_device_memory()

        #print(M_u_u.asarray())

        #update V
        up=manyDot([M_t_f, W, H2.transpose()]).add(cm.dot(S_t_t,V).mult(beta))
        psaiV= manyDot([V.transpose(), M_t_f, W, H2.transpose()]).subtract(manyDot([H2, W.transpose(), W, H2.transpose()])).subtract(manyDot([V.transpose(),L_t,V]).mult(beta))
        down= manyDot([V, H2, W.transpose(), W, H2.transpose()]).add(cm.dot(S_t_t_D,V).mult(beta)).add(cm.dot(V, psaiV))
        size=down.shape
        plus=np.ones(size)*pvalue
        plus=cm.CUDAMatrix(plus)
        down.add(plus)
        # print(down.asarray())
        # print(V.asarray())
        up.divide(down)
        up_cpu=up.asarray()
        up.free_device_memory()
        for i in range(0,len(up_cpu)):
            for j in range(0,len(up_cpu[i])):
                if up_cpu[i][j]<0:
                    up_cpu[i][j]=0
        up=cm.CUDAMatrix(up_cpu)
        V.mult(cm.sqrt(up))
        #print(V.asarray())

        up.free_device_memory()
        psaiV.free_device_memory()
        down.free_device_memory()
        plus.free_device_memory()


        #update W
        up= manyDot([M_t_f.transpose(), V, H2]).add(manyDot([M_u_f.transpose(), U, H3]))
        down= manyDot([W, H2.transpose(), V.transpose(), V, H2]).add(manyDot([W, H3.transpose(), U.transpose(), U, H3]))
        size=down.shape
        plus=np.ones(size)*pvalue
        plus=cm.CUDAMatrix(plus)
        down.add(plus)
        W.mult(cm.sqrt(up.divide(down)))

        up.free_device_memory()
        down.free_device_memory()
        plus.free_device_memory()

        #update H1
        up=manyDot([U.transpose(), M_u_u, U])
        down=manyDot([U.transpose(), U, H1, U.transpose(), U])
        size=down.shape
        plus=np.ones(size)*pvalue
        plus=cm.CUDAMatrix(plus)
        down.add(plus)
        #print(H1)
        H1.mult(cm.sqrt(up.divide(down)))
        #print(H1)

        up.free_device_memory()
        down.free_device_memory()
        plus.free_device_memory()

        #update H2
        up=manyDot([V.transpose(), M_t_f, W])
        down=manyDot([V.transpose(), V, H2, W.transpose(), W])
        size=down.shape
        plus=np.ones(size)*pvalue
        plus=cm.CUDAMatrix(plus)
        down.add(plus)
        H2.mult(cm.sqrt(up.divide(down)))

        up.free_device_memory()
        down.free_device_memory()
        plus.free_device_memory()

        #update H3
        up=manyDot([U.transpose(), M_u_f, W])
        down=manyDot([U.transpose(), U, H3, W.transpose(), W])
        size=down.shape
        plus=np.ones(size)*pvalue
        plus=cm.CUDAMatrix(plus)
        down.add(plus)
        H3.mult(cm.sqrt(up.divide(down)))

        up.free_device_memory()
        down.free_device_memory()
        plus.free_device_memory()


        step=step+1
        print('step: '+str(step))
        purity=per.dealWith(U.asarray())
        if purity>maxPurity:
            #print('ex')
            maxPurity=purity
            U_c=U.copy()
            maxU=U_c.asarray()
            U_c.free_device_memory()
            # maxU=U.asarray() maxU would keep up with U in gpu

    t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    #print(t)
    np.save('hashtag/U/U-'+t.replace(' ','-')+'.npy',maxU)
    print('Max Purity during this process: '+str(maxPurity))
    #per.dealWith(maxU)

def targetFunction(M_u_u,M_u_f,M_t_f,L_u,L_t,U,V,W,H1,H2,H3,alpha,beta):
    f1=forbenuis(M_u_u,U,H1,U)
    f2=forbenuis(M_t_f,V,H2,W)
    f3=forbenuis(M_u_f,U,H3,W)
    f1.add(f2).add(f3)

    t1=cm.dot(cm.dot(U.T,L_u),U)
    t1_cpu=t1.asarray()
    t1.free_device_memory()
    trace1=np.trace(t1_cpu)*alpha
    t2=cm.dot(cm.dot(V.T,L_t),V)
    t2_cpu=t2.asarray()
    t2.free_device_memory()
    trace2=np.trace(t2_cpu)*beta

    #print(f1.asarray())
    f1.add(trace1).add(trace2)
    #print(f1.asarray())
    return f1

def forbenuis(M,U,H,W):
    temp1=M.copy()
    temp1.subtract(cm.dot(cm.dot(U,H),W.transpose()))
    temp1.mult(temp1)
    #print(temp1.asarray())
    v1=temp1.sum(axis=0).sum(axis=1)
    #print(v1.asarray())
    return v1

def manyDot(arrayList):
    temp=arrayList[0]
    for i in range(1,len(arrayList)):
        temp=cm.dot(temp,arrayList[i])
    return temp

def test():
    M_u_u=np.arange(25).reshape(5,5)
    M_u_f=np.arange(20).reshape(5,4)
    M_t_f=np.arange(24).reshape(6,4)
    k=3
    loss=1
    num_step=10000
    optimization(M_u_u, M_u_f, M_t_f, k, loss, num_step)

def runHashTag():

    data_path='hashtag/Basing_AR_both_Part/'
    per.data_path=data_path

    M_u_u=np.load(data_path+'M_u_u.npy')
    M_u_f = np.load(data_path+'M_u_f_part.npy')
    # the same type and size as before save and load
    print('M_u_f size is: '+str(sys.getsizeof(M_u_f))+' byte')
    M_t_f = np.load(data_path+'M_t_f_part.npy')
    print('M_t_f size is: '+str(sys.getsizeof(M_t_f))+' byte')
    k=8
    loss=1
    num_step=5000
    S_u_u=np.load(data_path+'S_u_u_norm.npy')
    S_u_u_D=np.load(data_path+'S_u_u_D_norm.npy')
    L_u=np.load(data_path+'S_u_u_Laplace_norm.npy')
    S_t_t=np.load(data_path+'S_t_t_norm.npy')
    S_t_t_D=np.load(data_path+'S_t_t_D_norm.npy')
    L_t=np.load(data_path+'S_t_t_Laplace_norm.npy')
    alpha=0.5
    beta=0.5
    optimization(M_u_u,M_u_f,M_t_f,L_u,L_t,S_u_u,S_u_u_D,S_t_t,S_t_t_D,alpha,beta,k,loss,num_step)

def setup():
    cm.cublas_init()
def teardown():
    cm.cublas_shutdown()

def testPython():
    t=np.random.rand(2,2)
    t=cm.CUDAMatrix(t)
    nt=t.asarray()
    print(nt)
    s=t.sum(axis=0,)
    nt=s.asarray()
    print(nt)
    z=s.sum(axis=1)
    nt=z.asarray()
    print(nt)
    x=np.random.rand(2,2)
    print(x)
    x=cm.CUDAMatrix(x)
    m=t.mult(x)
    nm=m.asarray()
    print(nm)



setup()
#testPython()
#test()
runHashTag()
teardown()

# x=np.random.rand(2,2)
# print x
# print np.trace(x)
