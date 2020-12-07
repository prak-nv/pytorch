import unittest
import os
import random

import torch

from torch.testing._internal.common_utils import run_tests, ProfilingMode, GRAPH_EXECUTOR
from torch.testing._internal.codegen.random_topo_test import runDefaultTestWithSeed
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA

import itertools
import numpy as np
import math

os.environ['PYTORCH_NVFUSER_DISABLE_FALLBACK'] = '1'
os.environ['PYTORCH_NVFUSER_DISABLE_FMA'] = '1'
os.environ['PYTORCH_NVFUSER_JIT_OPT_LEVEL'] = '0'

if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)

FUSION_GROUP = 'prim::CudaFusionGroup'
FUSION_GUARD = 'prim::CudaFusionGuard'

# test suits for generating nvprof log with nvtx marker
# there is a separate test suit for parsing and comparing result
class TestBenchmarkNorm(JitTestCase):
    def setUp(self):
        super(TestBenchmarkNorm, self).setUp()
        self.old_cpu_fuse = torch._C._jit_can_fuse_on_cpu()
        self.old_gpu_fuse = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        self.old_guard = torch._C._jit_set_nvfuser_guard_mode(False)

        if(RUN_CUDA):
            self.old_nvfuser = torch._C._jit_set_nvfuser_enabled(True)

    def tearDown(self):
        if(RUN_CUDA):
            torch._C._jit_set_nvfuser_enabled(self.old_nvfuser)
        torch._C._jit_override_can_fuse_on_cpu(self.old_cpu_fuse)
        torch._C._jit_override_can_fuse_on_gpu(self.old_gpu_fuse)
        torch._C._jit_set_nvfuser_guard_mode(self.old_guard)
        super(TestBenchmarkNorm, self).tearDown()

    def _benchmark_helper(self, inputs, test_name, layer):
        # fix to 1000 iteration to get consistent result
        # parser can make same assumption to generate correct kernel time
        for i in range(1000):
            torch.cuda.nvtx.range_push(test_name+'_benchmarknvtx')
            o = layer(*inputs)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

    def _softmax_benchmark_helper(self, shape, reduction_axis, dtype, device):
        class MyLogSoftmax(torch.nn.Module):
            __constants__ = ['reduction_axis']

            def __init__(self):
                super(MyLogSoftmax, self).__init__()
                self.reduction_axis = reduction_axis

            def forward(self, x: torch.Tensor):
                o = torch.nn.functional.softmax(x, dim=self.reduction_axis)
                return torch.log(o)

        t = MyLogSoftmax()
        jit_t = torch.jit.script(t)
        # for softmax test, we use logsoftmax to get better comparison
        log_t = torch.nn.LogSoftmax(dim=reduction_axis)

        # create input and run one time to trigger compile
        x = torch.randn(shape, dtype=dtype, device=device)
        jit_o = jit_t(x)

        # create name for the test
        shape_str = ''
        for s in shape:
            shape_str += '_' + str(s)
        test_name = 'softmax' + shape_str + '_' + str(dtype).split(".")[1]

        # run both test with benchmark marker
        self._benchmark_helper([x], test_name+'_native', log_t)
        self._benchmark_helper([x], test_name+'_jit', jit_t)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_benchmark_softmax(self):
        for i in [512, 1024, 4096, 50000]:
            for j in [32, 64, 128, 1024, 2048]:
                self._softmax_benchmark_helper([i, j], 1, torch.float32, "cuda")
                self._softmax_benchmark_helper([i, j], 1, torch.float16, "cuda")

    def _batch_norm_benchmark_helper(self, shape, dtype, device):
        class MyBatchNorm(torch.nn.Module):
            def __init__(self):
                super(MyBatchNorm, self).__init__()

            def forward(self, x: torch.Tensor, r_mean: torch.Tensor, r_var: torch.Tensor):
                o = torch.nn.functional.batch_norm(x, r_mean, r_var, training=True)
                return torch.log(o)

        t = MyBatchNorm()
        jit_t = torch.jit.script(t)

        # create input and run one time to trigger compile
        x = torch.randn(shape, dtype=dtype, device=device)
        running_mean = torch.randn(shape[1], dtype=torch.float32, device=device)
        running_var = torch.randn(shape[1], dtype=torch.float32, device=device)
        jit_o = jit_t(x, running_mean.clone(), running_var.clone())

        eager_running_mean = running_mean.clone()
        eager_running_var = running_var.clone()
        jit_running_mean = running_mean.clone()
        jit_running_var = running_var.clone()

        # create name for the test
        shape_str = ''
        for s in shape:
            shape_str += '_' + str(s)
        test_name = 'batchnorm' + shape_str + '_' + str(dtype).split(".")[1]

        # run both test with benchmark marker
        self._benchmark_helper([x, eager_running_mean, eager_running_var], test_name+'_native', t)
        self._benchmark_helper([x, jit_running_mean, jit_running_var], test_name+'_jit', jit_t)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_benchmark_batch_norm(self):
        self._batch_norm_benchmark_helper([34,128,14,14], torch.float32, "cuda")
        self._batch_norm_benchmark_helper([34,1024,14,14], torch.float16, "cuda")


if __name__ == '__main__':
    run_tests()
