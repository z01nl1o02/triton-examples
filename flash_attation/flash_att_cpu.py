import torch
def online_softmax_torch(M, num_block=2):
    """
    online softmax: main aim is to get maximum and sum-exp of each column with in one scan from left to right
    """
    
    H,W = M.shape
    assert H%num_block == 0
    block_h = H // num_block
    for iblock in range(num_block):
        h0, h1 = iblock * block_h, (iblock+1) * block_h
        m = torch.zeros((block_h,),dtype=torch.float32) + torch.finfo(torch.float32).min
        d = torch.zeros((block_h,),dtype=torch.float32)
        for k in range(W):
            xk = M[h0:h1,k]
            m1 = torch.maximum(m, xk)
            d = d * torch.exp(m - m1) + torch.exp(xk - m1)
            m = m1
        M[h0:h1,:] = torch.exp(M[h0:h1,:] - m[:,None]) / d[:,None]
    return M

def test_online_softmax():
    M = torch.rand(18,10)
    gt = torch.nn.functional.softmax(M,dim=-1)
    te = online_softmax_torch(M, num_block=6)
    if torch.allclose(gt, te, atol=0.01, rtol=0.01):
        print(M.shape, " correct!")
    else:
        print(M.shape, " error!")


def bmm_torch(Q, K, BLOCK_M = 4, BLOCK_N = 4):
    """
    BLOCK_M/BLOCK_N: result height/width
    """
    
    assert Q.shape == K.shape
    B,H, N_CTX, _ = Q.shape
    assert N_CTX % BLOCK_M == 0 and N_CTX % BLOCK_N == 0
    out = torch.zeros(B,H,N_CTX, N_CTX)
    
    for ibh in range(0, B*H):
        ib, ih = ibh // H, ibh % H
        for im in range(0, N_CTX, BLOCK_M):
            blockq = Q[ib, ih, im:im+BLOCK_M]
            for inctx in range(0, N_CTX, BLOCK_N):
                blockk = K[ib, ih, inctx:inctx+BLOCK_N]
                qkt = blockq@torch.transpose(blockk,-1,-2)
                out[ib, ih, im:im+BLOCK_M, inctx:inctx+BLOCK_N] = qkt
    return out


        
def mha_torch(Q, K, V, BLOCK_M = 4, BLOCK_N = 4):
    """
    BLOCK_M/BLOCK_N: QK's height/width
    
    s1. get attation(=Q@A) with standard tiling
    s2. call online-softmax on each attation block
    s3. get output(=attation@V) by summary each attation@V block (attation is still in online softmax) 
    s4. rescale output with oneline softmax at the end of column. this is the exact result of MHA
    """
    assert Q.shape == K.shape == V.shape
    B, H, N_CTX, D = Q.shape
    assert N_CTX % BLOCK_M == 0 and N_CTX % BLOCK_N == 0
    O = torch.zeros_like(Q)
    for ibh in range(0, B*H):
        ib, ih = ibh // H, ibh % H
        for im in range(0, N_CTX, BLOCK_M):
            # thread start
            blockq = Q[ib, ih, im:im+BLOCK_M]
            acc = torch.zeros(BLOCK_M, D)
            m = torch.zeros((BLOCK_M,),dtype=torch.float32) + torch.finfo(torch.float32).min
            d = torch.zeros((BLOCK_M,),dtype=torch.float32)
            #import ipdb; ipdb.set_trace()
            for inctx in range(0, N_CTX, BLOCK_N):
                blockk = K[ib, ih, inctx:inctx+BLOCK_N]
                blockv = V[ib, ih, inctx:inctx+BLOCK_N]
                qkt = blockq@torch.transpose(blockk,-1,-2)
                m1 = torch.maximum(m, qkt.max(dim=-1)[0])
                # key: use m1/m with explict shape to aovid error when BLOCK_M == BLOCK_N 
                d =  d * torch.exp(m - m1) + torch.exp(qkt - m1[:,None]).sum(dim=-1)
                # running QKV!!!!
                acc = acc * torch.exp(m-m1)[:,None] + torch.exp(qkt - m1[:,None]) @ blockv
                m = m1
            acc = acc / d[:,None] # no substraction only division
            O[ib,ih,im:im+BLOCK_M] = acc 
    return O

def test_mha_torch():
    #Q = torch.rand(64,32,16,8)
    Q = torch.rand(4,4,4,4)
    K, V = torch.rand_like(Q), torch.rand_like(Q)
    gt = torch.nn.functional.softmax(torch.matmul(Q,K.transpose(-1,-2)),dim=-1)@V
    #gt = torch.matmul(Q,K.transpose(-1,-2))@V
    te = mha_torch(Q,K,V)
    if torch.allclose(gt, te):
        print(f"mha_torch: {tuple(Q.shape)},{tuple(K.shape)},{tuple(V.shape)} correct!!")
    else:
        print(f"mha_torch: {tuple(Q.shape)},{tuple(K.shape)},{tuple(V.shape)} error!!")
    #print(gt)
    #print("="*80)
    #print(te)

if __name__ == "__main__":
    #test_online_softmax()
    test_mha_torch()
