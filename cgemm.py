import os
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
import torch
import triton
import triton.language as tl

print(f"triton version: {triton.__version__}")
@triton.jit
def vec_mult_kernel001(A, B, C, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
    pid = tl.program_id(axis=0)
    y = pid // N
    x = pid % N
    a_offset = y * stride_am + tl.arange(0, K)
    b_offset = x * stride_bn + tl.arange(0, K) * stride_bk
    a_vec = tl.load(A+a_offset)
    b_vec = tl.load(B+b_offset)
    c = tl.sum(a_vec*b_vec,axis=0)
    tl.store(C + y * stride_cm + x * stride_cn, c)

def mat_mult001(A: torch.Tensor, B: torch.Tensor, M: int, K: int, N: int):
    C = torch.zeros((M,N), device="cuda")
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    grid = triton.cdiv( M*N, 1),
    vec_mult_kernel001[grid](A.data_ptr(), B.data_ptr(), C.data_ptr(), M, K, N, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn)
    torch.cuda.synchronize()
    return C

@triton.autotune(
    configs = [
        triton.Config(
            {
                "BLOCK_M": 32, 
                "BLOCK_K": 64,
                "BLOCK_N": 32,
            },
            num_stages=3,
            num_warps=8
        ),
        triton.Config(
            {
                "BLOCK_M": 64, 
                "BLOCK_K": 32,
                "BLOCK_N": 64,
            },
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {
                "BLOCK_M": 128, 
                "BLOCK_K": 16,
                "BLOCK_N": 128,
            },
            num_stages=4,
            num_warps=4
        )
    ],
    key=["M","K","N"]
)
@triton.jit
def vec_mult_kernel002(A, B, C, 
                       M, K, N, 
                       stride_am, stride_ak, 
                       stride_bk, stride_bn, 
                       stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
                       ):

    A = A.to(tl.pointer_type(tl.float32))
    B = B.to(tl.pointer_type(tl.float32))
    C = C.to(tl.pointer_type(tl.float32))

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)


    block_m = pid_m * BLOCK_M
    block_n = pid_n * BLOCK_N
   
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K): ### !!!K % BLOCK_K == 0

        A_offset_m = block_m + tl.arange(0,BLOCK_M)[:,None]
        A_offset_k = k + tl.arange(0, BLOCK_K)[None,:]
        A_offset = A_offset_m * stride_am + A_offset_k * stride_ak
        A_mask = (A_offset_m < M) & (A_offset_k < K)

        B_offset_k = k + tl.arange(0, BLOCK_K)[:,None]
        B_offset_n = block_n + tl.arange(0, BLOCK_N)[None,:]
        B_offset = B_offset_k * stride_bk + B_offset_n * stride_bn
        B_mask = (B_offset_k < K) & (B_offset_n < N)


        A_block = tl.load(A + A_offset, mask=A_mask, other=0.0)
        B_block = tl.load(B + B_offset, mask=B_mask, other=0.0)
        acc += tl.dot(A_block, B_block)
    
    C_offset_m = block_m + tl.arange(0, BLOCK_M)[:, None]
    C_offset_n = block_n + tl.arange(0, BLOCK_N)[None, :]
    C_offset = C_offset_m * stride_cm + C_offset_n * stride_cn
    C_mask = (C_offset_m < M) & (C_offset_n < N)
    tl.store(C + C_offset, acc, mask=C_mask)

def mat_mult002(A: torch.Tensor, B: torch.Tensor, M: int, K: int, N: int):
    C = torch.zeros((M,N), device="cuda")
    A = A.contiguous().float()
    B = B.contiguous().float()
    C = C.contiguous().float()
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    #BLOCK_M, BLOCK_K, BLOCK_N = 32, 16, 64
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    vec_mult_kernel002[grid](A.data_ptr(), B.data_ptr(), C.data_ptr(), M, K, N, 
                             stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn
                             #BLOCK_M, BLOCK_K, BLOCK_N
                             )
    torch.cuda.synchronize()
    #print(f"best config: {vec_mult_kernel002.best_config}")
    return C

testing_functions = {
    1:mat_mult001, 
    2:mat_mult002
}


def check_precision(kernel_id):
    M, K, N = 256, 128, 512
    A = torch.randn((M,K), device="cuda")
    B = torch.randn((K,N), device="cuda")
    C = testing_functions[kernel_id](A, B, M, K, N)
    Cgt = torch.matmul(A,B)
    if torch.allclose(C, Cgt, rtol=1e-2):
        print(f"precsion check OK! kernel_id = {kernel_id}")
    else:
        print(f"{C} ===\n==={Cgt}")
        print(f"precision check FAILED! kernel_id = {kernel_id}")

def check_performance(kernel_id):
    M, K, N = 256, 128, 512
    A = torch.randn((M,K), device="cuda")
    B = torch.randn((K,N), device="cuda")
    t_baseline = triton.testing.do_bench(lambda : torch.matmul(A, B), warmup=10, rep=100)
    if isinstance(t_baseline,(tuple, list)):
        t_baseline = t_baseline[0]
    func = lambda : testing_functions[kernel_id](A, B, M, K, N)
    t_custom = triton.testing.do_bench(func, warmup=10, rep=100) 
    if isinstance(t_custom,(tuple, list)):
        t_custom = t_custom[0]
    print(f"performance: baseline {t_baseline:.5f} ms, custom kernel {kernel_id} {t_custom:.5f} ms")

check_precision(2)
check_performance(2)

