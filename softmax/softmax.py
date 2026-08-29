import torch
import triton
import triton.language as tl
import triton.testing as tt

print(f"triton version {triton.__version__}")

def softmax_torch_slow(A):
    normA = torch.exp(A - A.max(dim=-1)[0][:,None])
    #print(f"max_each_col = {A.max(dim=-1)[0]}")
    #print(f"sum_each_col = {torch.sum(normA,dim=-1)}")
    res =  normA / torch.sum(normA,dim=-1)[:,None]
    return res

def softmax_torch(A):
    return torch.nn.functional.softmax(A,dim=-1)


#############################################################

@triton.jit
def _calc_max_kernel(input_ptr, M:int, N:int, stride_m:int, stride_n:int, output_ptr, stride_output:int, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    pn, pm = tl.program_id(0), tl.program_id(1)

    block_n = pn * BLOCK_N
    block_m = pm * BLOCK_M

    offset_n = block_n + tl.arange(0, BLOCK_N)[None,:]
    mask_n = offset_n < N
    offset_m = block_m + tl.arange(0, BLOCK_M)[:,None]
    mask_m = offset_m < M

    offset = offset_m * stride_m + offset_n * stride_n
    mask = mask_m & mask_n

    block_ptr = tl.load(input_ptr+offset, mask=mask, other=float('-inf'))

    max_value = tl.max(block_ptr, axis=1)
    
    offset_out = block_m + tl.arange(0,BLOCK_M)
    mask_out = offset_out < M
    tl.atomic_max(output_ptr + offset_out * stride_output, max_value, mask=mask_out)

def calc_max(A: torch.Tensor):
    M, N = A.size()
    A = A.contiguous()
    output = torch.zeros((M,),device="cuda") + torch.finfo(torch.float32).min
    output = output.contiguous()
    stride_m, stride_n = A.stride()
    stride_output = output.stride(0)
    BLOCK_N, BLOCK_M = 256, 128
    grid = triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M)
    _calc_max_kernel[grid](A.data_ptr(), M, N, stride_m, stride_n, output.data_ptr(), stride_output,  BLOCK_N, BLOCK_M)
    return output


def check_precision_max():
    M, N = 1024, 512
    A = torch.randn((M,N),device="cuda") * 100
    test = calc_max(A)
    gt = torch.max(A,dim=-1)[0]
    if torch.allclose(test, gt):
        print("max_kernel OK")
    else:
        print(f"max_kernel ERROR:  {test} != {gt} ")



@triton.jit
def _calc_sum_exp_sub_kernel(input_ptr, M:int, N:int, stride_m:int, stride_n:int, max_value_ptr, stride_max_value:int, output_ptr, stride_output:int, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pn, pm = tl.program_id(0), tl.program_id(1)

    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    max_value_ptr = max_value_ptr.to(tl.pointer_type(tl.float32))

    block_n = pn * BLOCK_N
    block_m = pm * BLOCK_M

    offset_n = block_n + tl.arange(0, BLOCK_N)[None,:]
    mask_n = offset_n < N
    offset_m = block_m + tl.arange(0, BLOCK_M)[:,None]
    mask_m = offset_m < M

    offset = offset_m * stride_m + offset_n * stride_n
    mask = mask_m & mask_n

    block_data = tl.load(input_ptr+offset, mask=mask, other=float('-inf'))

    offset_max_value = block_m + tl.arange(0, BLOCK_M)[:,None]
    mask_max_value = offset_max_value < M
    max_value_data = tl.load(max_value_ptr + offset_max_value * stride_max_value, mask=mask_max_value, other=0)
    block_data = tl.exp(block_data - max_value_data)
    sum_value = tl.sum(block_data,axis=1)

    offset_output = block_m + tl.arange(0, BLOCK_M)
    mask_output = offset_output < M
    tl.atomic_add(output_ptr + offset_output * stride_output, sum_value, mask=mask_output)

    # write in place
    tl.store(input_ptr + offset, block_data, mask=mask)

def calc_sum_exp_sub(A: torch.Tensor, max_value: torch.Tensor):
    M, N = A.size()
    A = A.contiguous()
    output = torch.zeros((M,),device="cuda")
    output = output.contiguous()
    max_value = max_value.contiguous()
    stride_m, stride_n = A.stride()
    stride_output = output.stride(0)
    stride_max_value = max_value.stride(0)
    BLOCK_N, BLOCK_M = 256, 128
    grid = triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M)
    _calc_sum_exp_sub_kernel[grid](A.data_ptr(), M, N, stride_m, stride_n, max_value.data_ptr(), stride_max_value, output.data_ptr(), stride_output,  BLOCK_N, BLOCK_M)
    return output


def check_precision_sum():
    M, N = 1024, 512
    A = torch.randn((M,N),device="cuda") 
    max_value = torch.max(A, dim=-1)[0]
    gt = torch.sum(torch.exp(A - max_value[:,None]),dim=-1)
    test = calc_sum_exp_sub(A, max_value)
    if torch.allclose(test, gt):
        print("sum_kernel OK")
    else:
        print(f"sum_kernel ERROR:  {test} should be equal to {gt} ")



@triton.jit
def _calc_div_kernel(input_ptr, M:int, N:int, stride_m:int, stride_n: int, sum_values_ptr, stride_sum_values: int, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pn, pm = tl.program_id(0), tl.program_id(1)

    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    sum_values_ptr = sum_values_ptr.to(tl.pointer_type(tl.float32))

    block_m = pm * BLOCK_M
    block_n = pn * BLOCK_N

    offset_m = block_m  + tl.arange(0, BLOCK_M)[:,None]
    mask_m = offset_m < M
    offset_n = block_n + tl.arange(0, BLOCK_N)[None,:]
    mask_n = offset_n < N

    offset = offset_m * stride_m + offset_n * stride_n
    mask = mask_m & mask_n
    block_data = tl.load(input_ptr + offset, mask = mask, other=0.0)

    offset_sum_value = block_m + tl.arange(0, BLOCK_M)[:,None]
    mask_sum_value = offset_sum_value < M
    sum_values_data = tl.load(sum_values_ptr + offset_sum_value * stride_sum_values, mask = mask_sum_value, other = 1.0)

    res_data = block_data / sum_values_data
    tl.store(input_ptr + offset,  res_data, mask=mask)

def calc_div(A: torch.Tensor, sum_value: torch.Tensor):
    M, N = A.size()
    A = A.contiguous()
    sum_value = sum_value.contiguous()
    stride_m, stride_n = A.stride()
    stride_sum_value = sum_value.stride(0)
    BLOCK_N, BLOCK_M = 256, 128
    grid = triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M)
    _calc_div_kernel[grid](A.data_ptr(), M, N, stride_m, stride_n, sum_value.data_ptr(), stride_sum_value, BLOCK_N, BLOCK_M)
    return A


def check_precision_div():
    M, N = 1024, 512
    A = torch.randn((M,N),device="cuda") * 100
    sum_value = torch.abs(torch.randn((M,), device="cuda")) + 10.0

    gt = A / sum_value[:,None]
    test = calc_div(A, sum_value)

    if torch.allclose(test, gt):
        print("div_kernel OK")
    else:
        print(f"div_kernel ERROR:  {Amax} - {AmaxGT} ")


def softmax_tiling(logist: torch.Tensor, block_m=2, block_n=2):
    M, N = logist.size()
    logist = logist.contiguous()
    stride_m, stride_n = logist.stride()

    BLOCK_N, BLOCK_M = block_n, block_m
    grid = triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M)

    max_each_col = torch.zeros((M,), device=logist.device).contiguous()
    stride = max_each_col.stride(0)
    _calc_max_kernel[grid](logist.data_ptr(), M, N, stride_m, stride_n, max_each_col.data_ptr(), stride,  BLOCK_N, BLOCK_M)
    sum_each_col = torch.zeros((M,), device=logist.device).contiguous()
    stride = sum_each_col.stride(0)
    _calc_sum_exp_sub_kernel[grid](logist.data_ptr(), M, N, stride_m, stride_n, max_each_col.data_ptr(), stride, sum_each_col.data_ptr(), stride,  BLOCK_N, BLOCK_M)

    _calc_div_kernel[grid](logist.data_ptr(), M, N, stride_m, stride_n, sum_each_col.data_ptr(), stride, BLOCK_N, BLOCK_M)
    return logist
##########################################################################################
@triton.jit
def _softmax_line_kernel(A_ptr, M:int, N:int, stride_m: int, stride_n: int, BLOCK_N:tl.constexpr, BLOCK_M:tl.constexpr):
    # !assert BLOCK_N == N
    pn, pm = tl.program_id(0), tl.program_id(1)
    A_ptr = A_ptr.to(tl.pointer_type(tl.float32))

    block_m, block_n = pm * BLOCK_M, pn * BLOCK_N

    offset_m = block_m + tl.arange(0, BLOCK_M)[:,None]
    offset_n = block_n + tl.arange(0, BLOCK_N)[None,:]

    offset = offset_m * stride_m + offset_n * stride_n
    mask = (offset_m < M) & (offset_n < N)

    A_data = tl.load(A_ptr + offset, mask = mask, other=-float("inf"))
    max_each_line = tl.max(A_data, axis=1)
    A_data = tl.exp(A_data - max_each_line[:,None])
    sum_each_line = tl.sum(A_data, axis=1)
    A_data = A_data / sum_each_line[:,None]

    tl.store(A_ptr + offset, A_data, mask = mask)

def softmax_line(A, block_m=64):
    A = A.contiguous()
    M,N = A.size()
    stride_m, stride_n = A.stride()

    BLOCK_M = block_m
    BLOCK_N = N
    grid = triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M)
    _softmax_line_kernel[grid](A.data_ptr(), M, N, stride_m, stride_n, BLOCK_N, BLOCK_M)
    return A

def check_precision_softmax_line():
    M, N = 1024, 512
    A = torch.randn((M,N),device="cuda") * 1
    gt =  softmax_torch(A)
    test = softmax_line(A)
    if torch.allclose(test, gt):
        print("softmax_line OK")
    else:
        print(f"softmax_line ERROR:  {test} - {gt} ")



##########################################################################################
def check_precision_softmax_tiling():
    M, N = 1024, 512
    A = torch.randn((M,N),device="cuda") * 1
    gt =  softmax_torch(A)
    test = softmax_tiling(A)
    if torch.allclose(test, gt):
        print("softmax_tiling OK")
    else:
        print(f"softmax_tiling ERROR:  {test} - {gt} ")

# https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#sphx-glr-getting-started-tutorials-02-fused-softmax-py

@tt.perf_report(
        tt.Benchmark(
            x_names = ["M"],
            x_vals=[256 * i for i in range(2,64)],
            line_arg="provider",
            line_vals=['gt_slow', 'gt', "test_tiling", "test_line"],
            line_names=['gt_slow', 'gt', "test_tiling", "test_line"],
            ylabel="GB/s",
            plot_name="performance",
            args={"N":4096}
            )
        )
def benchmark_softmax(M, N, provider):
    A = torch.randn((M,N),device="cuda") 

    if provider == "gt_slow":
        fn = lambda: softmax_torch_slow(A)
    if provider == "gt":
        fn = lambda: softmax_torch(A)
    if provider == "test_tiling":
        fn = lambda: softmax_tiling(A, block_m=8, block_n=8)
    if provider == "test_line":
        fn = lambda: softmax_line(A, block_m=2)

    ms = tt.do_bench(fn)
    calc_gbps = lambda ms: 2 * A.numel() * A.element_size() * 1e-9 / (ms * 1e-3)
    return calc_gbps(ms)



check_precision_softmax_tiling()
check_precision_softmax_line()

benchmark_softmax.run(show_plots=False, print_data=True, save_path="./")


