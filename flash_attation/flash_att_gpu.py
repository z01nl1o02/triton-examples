import torch
import math
import triton
import triton.language as tl
import fire
########################################################
# baseline
def baseline(K,Q,V):
    B,T,D = K.size()
    att =  torch.nn.functional.softmax(torch.bmm(Q, K.transpose(-1,-2)) / math.sqrt(D), dim=-1)
    return torch.bmm(att, Q)

########################################################
# 001
@triton.jit
def _bmm(A_ptr, B_ptr, C_ptr, 
        B, M, N,
        sa_b, sa_m, sa_n,
        sb_b, sb_m, sb_n,
        sc_b, sc_m, sc_n, 
        BLOCK_SIZE: tl.constexpr):

    A_ptr = A_ptr.to(tl.pointer_type(tl.float32)) # B, M, N
    B_ptr = B_ptr.to(tl.pointer_type(tl.float32)) # B, N, M
    C_ptr = C_ptr.to(tl.pointer_type(tl.float32)) # B, M, M

    cpid_b = tl.program_id(axis=0)
    cpid_m = tl.program_id(axis=1)
    cpid_n = tl.program_id(axis=2)

    cblock_b = cpid_b * BLOCK_SIZE
    cblock_m = cpid_m * BLOCK_SIZE
    cblock_n = cpid_n * BLOCK_SIZE
   
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for n in range(0, N, BLOCK_SIZE): ### !!!N % BLOCK_SIZE == 0


        A_offset_b = cblock_b + tl.arange(0,BLOCK_SIZE)[:,None,None]
        A_offset_m = cblock_m + tl.arange(0,BLOCK_SIZE)[None,:,None]
        A_offset_n = n + tl.arange(0, BLOCK_SIZE)[None,None,:]
        A_offset = A_offset_b * sa_b + A_offset_m * sa_m + A_offset_n * sa_n
        A_mask = (A_offset_b < B) & (A_offset_m < M) & (A_offset_n < N)

        B_offset_b = cblock_b + tl.arange(0, BLOCK_SIZE)[:,None,None]
        B_offset_m = n + tl.arange(0, BLOCK_SIZE)[None,:,None]
        B_offset_n = cblock_n + tl.arange(0, BLOCK_SIZE)[None,None,:]
        B_offset = B_offset_b * sb_b + B_offset_m * sb_m + B_offset_n * sb_n
        B_mask = (B_offset_b < B) & (B_offset_m < N) & (B_offset_n < M)


        A_block = tl.load(A_ptr + A_offset, mask=A_mask, other=0.0)
        B_block = tl.load(B_ptr + B_offset, mask=B_mask, other=0.0)
        acc += tl.dot(A_block, B_block)
    
    C_offset_b = cblock_b + tl.arange(0, BLOCK_SIZE)[:,None,None]
    C_offset_m = cblock_m + tl.arange(0, BLOCK_SIZE)[None,:,None]
    C_offset_n = cblock_n + tl.arange(0, BLOCK_SIZE)[None,None,:]
    C_offset = C_offset_b * sc_b + C_offset_m * sc_m + C_offset_n * sc_n
    C_mask = (C_offset_b < B) & (C_offset_m < M) & (C_offset_n < M)
    tl.store(C_ptr + C_offset, acc, mask=C_mask)

def bmm(A: torch.Tensor, B: torch.Tensor, BLOCK_SIZE=16):
    batch, M, N = A.size()
    #assert B,N,M == B.size()
    C = torch.zeros((batch,M,M), device="cuda")
    A = A.contiguous().float()
    B = B.contiguous().float()
    C = C.contiguous().float()
    sa_b, sa_m, sa_n = A.stride()
    sb_b, sb_m, sb_n = B.stride()
    sc_b, sc_m, sc_n = C.stride()

    grid = triton.cdiv(batch, BLOCK_SIZE), triton.cdiv(M, BLOCK_SIZE), triton.cdiv(M, BLOCK_SIZE)
    _bmm[grid](A.data_ptr(), B.data_ptr(), C.data_ptr(), 
              batch, M, N, 
              sa_b, sa_m, sa_n,
              sb_b, sb_m, sb_n,
              sc_b, sc_m, sc_n, 
              BLOCK_SIZE)
    torch.cuda.synchronize()
    return C

def test_bmm_accuracy():
    B, T, D = 16*5, 16*4, 16*8
    device = torch.device("cuda:0")
    K = torch.randn((B,T,D),device=device)
    Q = torch.randn((B,T,D),device=device)
    gt = torch.bmm(K, Q.transpose(-1,-2))
    query = bmm(K, Q.transpose(-1,-2), BLOCK_SIZE=16)
    if torch.allclose(gt, query, rtol=1.0, atol=0.1):
        print("bmm accuracy test: passed")
    else:
        print("bmm accuracy test: failed")
        print("="*80)
        print(f"gt={gt}")
        print(f"query={query}")
        print(f"max atol = {torch.abs(gt-query).max()}")



def test_baseline():
    B, T, D = 4, 8, 16
    device = torch.device("cuda:0")
    K = torch.randn((B,T,D),device=device)
    V = torch.randn((B,T,D),device=device)
    Q = torch.randn((B,T,D),device=device)
    baseline_result = baseline(K,Q,V)
    print(f"B,T,D = {B},{T},{D}")
    print("="*80)
    print(f"baseline: {baseline_result.size()}")


###########################################################################################
def online_softmax_torch(M, num_block=2):
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

################################################################################

@triton.jit
def _mha_gpu_kernel(qbase, kbase, vbase, obase,
        qstride0, qstride1, qstride2, qstride3,
        kstride0, kstride1, kstride2, kstride3,
        vstride0, vstride1, vstride2, vstride3,
        ostride0, ostride1, ostride2, ostride3,
        B, H, N_CTX, 
        D: tl.constexpr,
        BLOCK_M: tl.constexpr, 
        BLOCK_N: tl.constexpr):

    qbase = qbase.to(tl.pointer_type(tl.float32)) 
    kbase = kbase.to(tl.pointer_type(tl.float32)) 
    vbase = vbase.to(tl.pointer_type(tl.float32)) 
    obase = obase.to(tl.pointer_type(tl.float32))


    ibhn = tl.program_id(axis=0) * BLOCK_M
    ib, ihn = ibhn // (H*N_CTX), ibhn%(H*N_CTX)
    ih, im = ihn // N_CTX, ihn % N_CTX
    qo_offset_m = tl.arange(0, BLOCK_M)[:,None] + im
    qo_offset_d =  tl.arange(0, D)[None,:]
    qo_mask = (qo_offset_m < N_CTX)
    blockq_ptr = tl.load(qbase + ib * qstride0 + ih * qstride1 +  qo_offset_m * qstride2 + qo_offset_d * qstride3, 
            mask = qo_mask, other = 0.0)

    #tl.device_print("blockq.shape:",blockq_ptr.shape)
    #tl.device_print("blockq_ptr:",blockq_ptr)

    acc_ptr = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    #m = tl.zeros((BLOCK_M,), dtype=tl.float32) + tl.finfo(tl.float32).min
    m_ptr = tl.zeros((BLOCK_M,), dtype=tl.float32) + float("-inf") #(-3.4e+38)
    d_ptr = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for inctx in range(0, N_CTX, BLOCK_N):
        kv_offset_n, kv_offset_d = tl.arange(0,BLOCK_N)[:,None], tl.arange(0,D)[None,:]
        kv_mask = (kv_offset_n < N_CTX)
        blockk_ptr = tl.load(kbase + ib * kstride0 + ih * kstride1 + inctx*kstride2 + kv_offset_n * kstride2 + kv_offset_d * kstride3,
                mask = kv_mask, other = 0.0)
        blockv_ptr = tl.load(vbase + ib * vstride0 + ih * vstride1 + inctx*vstride2 + kv_offset_n * vstride2 + kv_offset_d * vstride3,
                mask = kv_mask, other = 0.0)


        kt_ptr = tl.trans(blockk_ptr)
        #tl.device_print("shape:",kt_ptr.shape)

        qkt_ptr = tl.dot(blockq_ptr, kt_ptr)
        m1_ptr = tl.maximum(m_ptr, tl.max(qkt_ptr,1))
        d_ptr = d_ptr * tl.math.exp(m_ptr - m1_ptr) + tl.sum(tl.math.exp(qkt_ptr - m1_ptr[:,None]),1)

        aa = tl.math.exp(qkt_ptr - m1_ptr[:,None])
        acc_ptr = acc_ptr * tl.math.exp(m_ptr - m1_ptr)[:,None] + tl.dot(aa, blockv_ptr)
        ####acc_ptr = acc_ptr * tl.math.exp(m_ptr - m1_ptr)[:,None] + tl.dot(tl.math.exp(qkt_ptr - m1_ptr[:,None]), blockv_ptr)
        m_ptr = m1_ptr
    acc_ptr = acc_ptr / d_ptr[:,None]
    #tl.device_print("acc_ptr:",acc_ptr)
    #tl.device_print("qo_mask shape1:",qo_mask.shape[1])
    blocko_ptr = obase + ib * ostride0 + ih * ostride1  + qo_offset_m * ostride2 + qo_offset_d * ostride3
    if 0:
        tl.device_print("acc_ptr:",acc_ptr)
        #tl.device_print("qo mask:",qo_mask)
        tl.device_print("qo_offset_m", qo_offset_m)
        #tl.device_print("stride: ", ostride0, ostride1,ostride2, ostride3)
    tl.store(blocko_ptr, acc_ptr.to(obase.dtype.element_ty),mask = qo_mask)
    return

def mha_gpu(Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor, BLOCK_M:int = 16, BLOCK_N:int = 16) -> torch.Tensor:
    assert Q.shape == K.shape == V.shape
    B, H, N_CTX, D = Q.shape
    O = torch.zeros_like(V)
    Q = Q.contiguous().float()
    K = K.contiguous().float()
    V = V.contiguous().float()
    O = O.contiguous().float()
    qstride0, qstride1, qstride2, qstride3 = Q.stride()
    kstride0, kstride1, kstride2, kstride3 = K.stride()
    vstride0, vstride1, vstride2, vstride3 = V.stride()
    ostride0, ostride1, ostride2, ostride3 = O.stride()
    assert N_CTX % BLOCK_M == N_CTX % BLOCK_N == 0
    #grid = triton.cdiv(B*H*N_CTX, BLOCK_M), triton.cdiv(N_CTX, BLOCK_N)
    grid = triton.cdiv(B*H*N_CTX, BLOCK_M),
    _mha_gpu_kernel[grid](
            Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(),
            qstride0, qstride1, qstride2, qstride3,
            kstride0, kstride1, kstride2, kstride3,
            vstride0, vstride1, vstride2, vstride3,
            ostride0, ostride1, ostride2, ostride3,
            B, H, N_CTX, D,
            BLOCK_M, BLOCK_N)
    torch.cuda.synchronize()   #it is useless for triton.testing.do_bench
    return O

def mha_torch(Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor) -> torch.Tensor:
    return  torch.nn.functional.softmax(torch.matmul(Q,K.transpose(-1,2)),dim=-1)@V

def checking_mha():
    device = torch.device("cuda:0")
    Q = torch.rand((3,20,5*16,16*4),device=device,dtype=torch.float32)
    K, V = torch.rand_like(Q), torch.rand_like(Q)
    gt = mha_torch(Q,K,V)
    #gt = torch.nn.functional.softmax(torch.matmul(Q,K.transpose(-1,2)),dim=-1)@V
    te = mha_gpu(Q,K,V)
    if torch.allclose(gt,te, atol=1e-3):
        print(f"mha_torch: {tuple(Q.shape)},{tuple(K.shape)},{tuple(V.shape)} correct!!")
    else:
        print(f"mha_torch: {tuple(Q.shape)},{tuple(K.shape)},{tuple(V.shape)} error!!")
        print(gt)
        print('----------')
        print(te)

BATCH_SIZE, HEAD_NUM = 4, 32
configs = []
for HEAD_DIM in [16,32,64]:
    for BLOCK_M in [16,32]:
        for BLOCK_N in [16,32]:
            configs.append(
                    triton.testing.Benchmark(
                        x_names=["N_CTX"],
                        x_vals = [2**i for i in range(5,10)],
                        line_arg="provider",
                        line_vals=["torch","triton"],
                        line_names=["torch","triton"],
                        styles=[("red",'-'),("blue",'-')],
                        ylabel="TFLOPS",
                        plot_name=f"MHA_BATCH(f{BATCH_SIZE})-HEAD({HEAD_NUM})-BLOCK_M({BLOCK_M})-BLOCK_N({BLOCK_N})",
                        args={
                            "B": BATCH_SIZE,
                            "H": HEAD_NUM,
                            "D": HEAD_DIM,
                            "BLOCK_M": BLOCK_M,
                            "BLOCK_N": BLOCK_N
                            }))

@triton.testing.perf_report(configs)
def profiling_mha(B,H,N_CTX,D, BLOCK_M=16,BLOCK_N=16, provider='torch'):
    device = torch.device("cuda:0")
    Q = torch.rand((B,H,N_CTX,D),device=device,dtype=torch.float32)
    K, V = torch.rand_like(Q), torch.rand_like(Q)
    if provider == "torch":
        fn = lambda: mha_torch(Q,K,V)
        ms = triton.testing.do_bench(fn)
    if provider == "triton":
        fn = lambda: mha_gpu(Q,K,V,BLOCK_M, BLOCK_N)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * B * H * N_CTX * N_CTX * D
    total_flops = 2 * flops_per_matmul
    return total_flops * 1e-12 / (ms * 1e-3)

if __name__ == "__main__":
    #checking_mha(),
    profiling_mha.run(save_path=".",print_data=True)
