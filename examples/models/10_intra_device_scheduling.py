# Copyright (c) PyPTO Contributors.
# -----------------------------------------------------------------------------------------------------------

import argparse
import torch
import pypto.language as pl
from pypto.runtime import RunConfig, run

SIZE = 32  
LOOPS = 8000

@pl.program
class UltimateHeftContentionProgram:
    
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_root(self, x: pl.Tensor[[SIZE, SIZE], pl.FP16], w: pl.Tensor[[SIZE, SIZE], pl.FP16], out: pl.Out[pl.Tensor[[SIZE, SIZE], pl.FP16]]) -> pl.Tensor[[SIZE, SIZE], pl.FP16]:
        x_l1 = pl.load(x, [0, 0], [SIZE, SIZE], target_memory=pl.MemorySpace.Mat)
        w_l1 = pl.load(w, [0, 0], [SIZE, SIZE], target_memory=pl.MemorySpace.Mat)
        x_l0a = pl.move(x_l1, target_memory=pl.MemorySpace.Left)
        w_l0b = pl.move(w_l1, target_memory=pl.MemorySpace.Right)
        
        acc = pl.cast(pl.matmul(x_l0a, w_l0b), pl.FP16)
        for _ in pl.range(LOOPS):
            tmp = pl.cast(pl.matmul(x_l0a, w_l0b), pl.FP16)
            
            acc = pl.add(acc, tmp)
            acc = pl.sub(acc, tmp)
            
        pl.store(acc, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_branch_a(self, t_root: pl.Tensor[[SIZE, SIZE], pl.FP16], w: pl.Tensor[[SIZE, SIZE], pl.FP16], out: pl.Out[pl.Tensor[[SIZE, SIZE], pl.FP16]]) -> pl.Tensor[[SIZE, SIZE], pl.FP16]:
        root_l1 = pl.load(t_root, [0, 0], [SIZE, SIZE], target_memory=pl.MemorySpace.Mat)
        w_l1 = pl.load(w, [0, 0], [SIZE, SIZE], target_memory=pl.MemorySpace.Mat)
        root_l0a = pl.move(root_l1, target_memory=pl.MemorySpace.Left)
        w_l0b = pl.move(w_l1, target_memory=pl.MemorySpace.Right)
        
        acc = pl.cast(pl.matmul(root_l0a, w_l0b), pl.FP16)
        for _ in pl.range(LOOPS):
            tmp = pl.cast(pl.matmul(root_l0a, w_l0b), pl.FP16)
            acc = pl.add(acc, tmp)
            acc = pl.sub(acc, tmp)
            
        pl.store(acc, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_branch_b(self, t_root: pl.Tensor[[SIZE, SIZE], pl.FP16], w: pl.Tensor[[SIZE, SIZE], pl.FP16], out: pl.Out[pl.Tensor[[SIZE, SIZE], pl.FP16]]) -> pl.Tensor[[SIZE, SIZE], pl.FP16]:
        root_l1 = pl.load(t_root, [0, 0], [SIZE, SIZE], target_memory=pl.MemorySpace.Mat)
        w_l1 = pl.load(w, [0, 0], [SIZE, SIZE], target_memory=pl.MemorySpace.Mat)
        root_l0a = pl.move(root_l1, target_memory=pl.MemorySpace.Left)
        w_l0b = pl.move(w_l1, target_memory=pl.MemorySpace.Right)
        
        acc = pl.cast(pl.matmul(root_l0a, w_l0b), pl.FP16)
        for _ in pl.range(LOOPS):
            tmp = pl.cast(pl.matmul(root_l0a, w_l0b), pl.FP16)
            acc = pl.add(acc, tmp)
            acc = pl.sub(acc, tmp)
            
        pl.store(acc, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_bg(self, x: pl.Tensor[[SIZE, SIZE], pl.FP16], w: pl.Tensor[[SIZE, SIZE], pl.FP16], out: pl.Out[pl.Tensor[[SIZE, SIZE], pl.FP16]]) -> pl.Tensor[[SIZE, SIZE], pl.FP16]:
        x_l1 = pl.load(x, [0, 0], [SIZE, SIZE], target_memory=pl.MemorySpace.Mat)
        w_l1 = pl.load(w, [0, 0], [SIZE, SIZE], target_memory=pl.MemorySpace.Mat)
        x_l0a = pl.move(x_l1, target_memory=pl.MemorySpace.Left)
        w_l0b = pl.move(w_l1, target_memory=pl.MemorySpace.Right)
        
        acc = pl.cast(pl.matmul(x_l0a, w_l0b), pl.FP16)
        for _ in pl.range(LOOPS):
            tmp = pl.cast(pl.matmul(x_l0a, w_l0b), pl.FP16)
            acc = pl.add(acc, tmp)
            acc = pl.sub(acc, tmp)
            
        pl.store(acc, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_merge(self, a: pl.Tensor[[SIZE, SIZE], pl.FP16], b: pl.Tensor[[SIZE, SIZE], pl.FP16], out: pl.Out[pl.Tensor[[SIZE, SIZE], pl.FP16]]) -> pl.Tensor[[SIZE, SIZE], pl.FP16]:
        a_l1 = pl.load(a, [0, 0], [SIZE, SIZE], target_memory=pl.MemorySpace.Mat)
        b_l1 = pl.load(b, [0, 0], [SIZE, SIZE], target_memory=pl.MemorySpace.Mat)
        a_l0a = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
        b_l0b = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
        
        acc = pl.cast(pl.matmul(a_l0a, b_l0b), pl.FP16)
        for _ in pl.range(LOOPS):
            tmp = pl.cast(pl.matmul(a_l0a, b_l0b), pl.FP16)
            acc = pl.add(acc, tmp)
            acc = pl.sub(acc, tmp)
            
        pl.store(acc, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def run_heft_dag(
        self, x: pl.Tensor[[SIZE, SIZE], pl.FP16], w: pl.Tensor[[SIZE, SIZE], pl.FP16],
        bg0: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]], bg1: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        bg2: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]], bg3: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        bg4: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]], bg5: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        bg6: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]], bg7: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        bg8: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]], bg9: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        bg10: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]], bg11: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        bg12: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]], bg13: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        bg14: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]], bg15: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        bg16: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]], bg17: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        bg18: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]], bg19: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        bg20: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]], bg21: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        bg22: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]], bg23: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        root_buf: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        a_buf: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        b_buf: pl.InOut[pl.Tensor[[SIZE, SIZE], pl.FP16]],
        out: pl.Out[pl.Tensor[[SIZE, SIZE], pl.FP16]]
    ):
        with pl.manual_scope():
            
            _, t_bg0 = pl.submit(self.kernel_bg, x, w, bg0)
            _, t_bg1 = pl.submit(self.kernel_bg, x, w, bg1)
            _, t_bg2 = pl.submit(self.kernel_bg, x, w, bg2)
            _, t_bg3 = pl.submit(self.kernel_bg, x, w, bg3)
            _, t_bg4 = pl.submit(self.kernel_bg, x, w, bg4)
            _, t_bg5 = pl.submit(self.kernel_bg, x, w, bg5)
            _, t_bg6 = pl.submit(self.kernel_bg, x, w, bg6)
            _, t_bg7 = pl.submit(self.kernel_bg, x, w, bg7)
            _, t_bg8 = pl.submit(self.kernel_bg, x, w, bg8)
            _, t_bg9 = pl.submit(self.kernel_bg, x, w, bg9)
            _, t_bg10 = pl.submit(self.kernel_bg, x, w, bg10)
            _, t_bg11 = pl.submit(self.kernel_bg, x, w, bg11)
            _, t_bg12 = pl.submit(self.kernel_bg, x, w, bg12)
            _, t_bg13 = pl.submit(self.kernel_bg, x, w, bg13)
            _, t_bg14 = pl.submit(self.kernel_bg, x, w, bg14)
            _, t_bg15 = pl.submit(self.kernel_bg, x, w, bg15)
            _, t_bg16 = pl.submit(self.kernel_bg, x, w, bg16)
            _, t_bg17 = pl.submit(self.kernel_bg, x, w, bg17)
            _, t_bg18 = pl.submit(self.kernel_bg, x, w, bg18)
            _, t_bg19 = pl.submit(self.kernel_bg, x, w, bg19)
            _, t_bg20 = pl.submit(self.kernel_bg, x, w, bg20)
            _, t_bg21 = pl.submit(self.kernel_bg, x, w, bg21)
            _, t_bg22 = pl.submit(self.kernel_bg, x, w, bg22)
            _, t_bg23 = pl.submit(self.kernel_bg, x, w, bg23)
            
            
            _, t_root = pl.submit(self.kernel_root, x, w, root_buf)
            _, t_branch_a = pl.submit(self.kernel_branch_a, root_buf, w, a_buf, deps=[t_root])
            _, t_branch_b = pl.submit(self.kernel_branch_b, root_buf, w, b_buf, deps=[t_root])
            _, t_merge = pl.submit(self.kernel_merge, a_buf, b_buf, out, deps=[t_branch_a, t_branch_b])
            
        return out

def heavy_op_pytorch(x_t: torch.Tensor, w_t: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x_t.float(), w_t.float()).half()

def verify_massive_heft_pytorch(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    t_root = heavy_op_pytorch(x, w)
    a = heavy_op_pytorch(t_root, w)
    b = heavy_op_pytorch(t_root, w)
    final_res = heavy_op_pytorch(a, b)
    return final_res

if __name__ == "__main__":
    cfg = RunConfig(platform="a2a3", enable_l2_swimlane=True, device_id=12)
    
    torch.manual_seed(42)
    x = torch.randn(SIZE, SIZE, dtype=torch.float16) / 100.0
    w = torch.randn(SIZE, SIZE, dtype=torch.float16) / 100.0
    
    bg_bufs = [torch.zeros(SIZE, SIZE, dtype=torch.float16) for _ in range(24)]
    root_buf = torch.zeros(SIZE, SIZE, dtype=torch.float16)
    a_buf = torch.zeros(SIZE, SIZE, dtype=torch.float16)
    b_buf = torch.zeros(SIZE, SIZE, dtype=torch.float16)
    
    out_pypto = torch.zeros(SIZE, SIZE, dtype=torch.float16)
    
    compiled = run(UltimateHeftContentionProgram, config=cfg)
    compiled(x, w, *bg_bufs, root_buf, a_buf, b_buf, out_pypto, config=cfg)
    
    out_ref = verify_massive_heft_pytorch(x, w)
    
    try:
        torch.testing.assert_close(
            out_pypto, out_ref, rtol=1e-2, atol=1e-2, 
            msg="Verification failed!"
        )
        print("✅ Success! PyPTO result is equal to Pytorch")
    except AssertionError as e:
        print("❌ Verification failed")
        print(e)