import unittest
import os
import random

import torch

from torch.testing._internal.common_utils import run_tests, ProfilingMode, GRAPH_EXECUTOR
from torch.testing._internal.codegen.random_topo_test import runDefaultTestWithSeed

from test_jit import JitTestCase, RUN_CUDA
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


class TestCudaFuser(JitTestCase):

    special_values = torch.tensor(
        [float("-inf"), -10, -math.pi,
            -1, -0.5, 0, 1, 0.5,
            math.pi, 10, float("inf"),
            float("nan")], dtype=torch.float, device='cuda')

    def _getSubgraphInFusion(self, graph):
        num_node = 0
        subgraph = None

        def count(block, ret):
            for n in block.nodes():
                if n.kind() == FUSION_GROUP:
                    ret[0] = ret[0] + 1
                    self.assertTrue(n.hasAttribute('Subgraph'))
                    ret[1] = n.g('Subgraph')
                for block in n.blocks():
                    count(block, ret)
        ret = [num_node, subgraph]
        count(graph, ret)
        self.assertEqual(ret[0], 1)
        return ret[1]

    def setUp(self):
        super(TestCudaFuser, self).setUp()
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
        super(TestCudaFuser, self).tearDown()

    def _run_helper(self, jit_op, op, *args):
        torch.cuda.manual_seed_all(123)
        jit_o = jit_op(*args)
        torch.cuda.manual_seed_all(123)
        jit_o = jit_op(*args)
        torch.cuda.manual_seed_all(123)
        o = op(*args)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(jit_op.graph_for(*args), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_half(self):
        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, alpha: float):
            o_16 = torch.add(x, y)
            o_32_a = torch.add(y, z, alpha=alpha)
            o_32_b = torch.add(o_16, z)
            return (o_16, o_32_a, o_32_b)

        t_jit = torch.jit.script(t)
        alpha = 0.5
        # stick to integers, this avoid the numerical difference due to our
        # promotion
        x = torch.randint(0, 256, (4, 8)).to(dtype=torch.float16, device="cuda")
        y = torch.randint(0, 256, (4, 8)).to(dtype=torch.float16, device="cuda")
        z = torch.randint(0, 256, (4, 8)).to(dtype=torch.float16, device="cuda")
        jit_o = t_jit(x, y, z, alpha)
        jit_o = t_jit(x, y, z, alpha)
        o = t(x, y, z, alpha)
        for oo, jit_oo in zip(o, jit_o):
            self.assertEqual(oo.dtype, jit_oo.dtype)
            self.assertEqual(oo, jit_oo)
        self.assertGraphContains(t_jit.graph_for(x, y, z, alpha), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_const(self):
        def t(x, y):
            o = x + y
            o = o + 2.0
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_chunk(self):
        def t(x, y, z, q):
            o = x + q
            x0, x1 = torch.chunk(o, 2)
            o = x0 + x1
            o = o + y
            o = o * z
            o = torch.relu(o)
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, dtype=torch.float, device="cuda")
        y = torch.randn(2, 8, dtype=torch.float, device="cuda")
        z = torch.randn(2, 8, dtype=torch.float, device="cuda")
        q = torch.randn(4, 8, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, z, q)
        jit_o = t_jit(x, y, z, q)
        o = t(x, y, z, q)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, z, q), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_reduction_double(self):
        def t(x: torch.Tensor):
            o = torch.mul(x, 1.0)
            o = torch.add(o, x)
            o = torch.sum(o, dim=[2], dtype=torch.double)
            return o
        t_jit = torch.jit.script(t)

        x = torch.randn(8, 4, 16, dtype=torch.double, device="cuda")
        jit_o = t_jit(x)
        jit_o = t_jit(x)
        o = t(x)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_scalar_input(self):
        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, 1, 32, dtype=torch.float, device="cuda")
        y = y.expand(4, 8, 32, 32)
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, 2.0), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_0(self):

        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        subgraph = self._getSubgraphInFusion(t_jit.graph_for(x, y, 2.0))
        self.assertGraphContainsExactly(subgraph, 'aten::add', 2, consider_subgraphs=False)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_1(self):

        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(1, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        subgraph = self._getSubgraphInFusion(t_jit.graph_for(x, y, 2.0))
        self.assertGraphContainsExactly(subgraph, 'aten::add', 2, consider_subgraphs=False)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_2(self):

        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 1, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(8, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        subgraph = self._getSubgraphInFusion(t_jit.graph_for(x, y, 2.0))
        self.assertGraphContainsExactly(subgraph, 'aten::add', 2, consider_subgraphs=False)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_3(self):

        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(8, 17, 8, dtype=torch.float, device="cuda")
        y = torch.randn(8, 17, 1, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        subgraph = self._getSubgraphInFusion(t_jit.graph_for(x, y, 2.0))
        self.assertGraphContainsExactly(subgraph, 'aten::add', 2, consider_subgraphs=False)

    # test_broadcasting_partition_logic_X
    # Testing partition logic that is capable to avoid creating unsupported
    # broadcasting semantics in CudaFusionGroup
    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_partition_logic_0(self):

        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            x = x + 12.0
            o1 = x + y
            o2 = x + z
            o = o1 + o2
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 6, 8, dtype=torch.float32, device="cuda")
        y = torch.randn(8, 6, 8, dtype=torch.float32, device="cuda")
        z = torch.randn(6, 8, dtype=torch.float32, device="cuda")
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        self.assertEqual(o, jit_o)
        subgraph = self._getSubgraphInFusion(t_jit.graph_for(x, y, z))
        self.assertGraphContainsExactly(subgraph, 'aten::add', 4, consider_subgraphs=False)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_partition_logic_1(self):

        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            x = x + 12.0
            o1 = x + y
            o2 = x + z
            o = o1 + o2
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(8, 6, 8, dtype=torch.float32, device="cuda")
        y = torch.randn(4, 8, 6, 8, dtype=torch.float32, device="cuda")
        z = torch.randn(4, 1, 6, 8, dtype=torch.float32, device="cuda")
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        self.assertEqual(o, jit_o)
        subgraph = self._getSubgraphInFusion(t_jit.graph_for(x, y, z))
        self.assertGraphContainsExactly(subgraph, 'aten::add', 2, consider_subgraphs=False)

    @unittest.skipIf(True, "Broadcast with different output not supported yet")
    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_multiple_output_shape(self):
        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = x + 12
            o1 = o + y
            o2 = o + z
            oo = o1.sum() + o2.sum()
            return oo
        t_jit = torch.jit.script(t)
        x = torch.randn(32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(2, 32, 32, dtype=torch.float, device="cuda")
        z = torch.randn(4, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        self.assertEqual(o, jit_o)
        # Currently cannot fuse this
        self.assertGraphContains(t_jit.graph_for(x, y, z), FUSION_GUARD)

    @unittest.skipIf(True, "broadcast on branches can't be resolved yet")
    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_multiple_output(self):
        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = x + 12
            o1 = o + y
            o2 = o + z
            oo = o1.sum() + o2.sum()
            return oo
        t_jit = torch.jit.script(t)
        x = torch.randn(32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 32, 32, dtype=torch.float, device="cuda")
        z = torch.randn(4, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        self.assertEqual(o, jit_o)
        # Currently cannot fuse this
        self.assertGraphContains(t_jit.graph_for(x, y, z), FUSION_GUARD)

    def _binary_test_helper(self, operation):
        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = x + z
            o = operation(o, y)
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, 2.0), FUSION_GUARD)

    def _unary_test_helper(self, operation):
        def t(x: torch.Tensor, z: float):
            o = x + z
            o = operation(o)
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, 2.0)
        jit_o = t_jit(x, 2.0)
        o = t(x, 2.0)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, 2.0), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_unary_ops(self):
        operations = [torch.neg,
                      torch.abs,
                      torch.log,
                      torch.log10,
                      torch.log1p,
                      torch.log2,
                      torch.lgamma,
                      torch.exp,
                      torch.expm1,
                      torch.erf,
                      torch.erfc,
                      torch.cos,
                      torch.acos,
                      torch.cosh,
                      torch.sin,
                      torch.asin,
                      torch.tan,
                      torch.atan,
                      torch.sqrt,
                      torch.rsqrt,
                      torch.ceil,
                      torch.floor,
                      torch.round,
                      torch.trunc,
                      torch.frac,
                      torch.reciprocal,
                      torch.relu,
                      torch.sigmoid,
                      torch.tanh,
                      torch.nn.functional.gelu]
        for op in operations:
            self._unary_test_helper(op)

    def _unary_type_test_helper(self, operation, dtype, data=None):
        shape = (4, 8, 32, 32)

        def t(x: torch.Tensor):
            o = x * 1.0
            o = operation(o)
            return o

        try:
            if data is None:
                x = torch.randn(shape, dtype=dtype, device="cuda")
            else:
                x = special_values.to(dtype=dtype)
            ref = t(x)
        except Exception:
            # same way as TE checker, if eager mode throws, ignore this test
            return
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x)
        jit_o = t_jit(x)
        o = t(x)
        self.assertEqual(o, jit_o, msg=f"""
        failing case:
            {dtype} {operation} {data}
        """)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_data_compatibility(self):
        dtypes = [
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bool
        ]
        operations = [torch.neg,
                      torch.abs,
                      torch.log,
                      torch.log10,
                      torch.log1p,
                      torch.log2,
                      torch.lgamma,
                      torch.exp,
                      torch.expm1,
                      torch.erf,
                      torch.erfc,
                      torch.cos,
                      torch.acos,
                      torch.cosh,
                      torch.sin,
                      torch.asin,
                      torch.tan,
                      torch.atan,
                      torch.sqrt,
                      torch.rsqrt,
                      torch.ceil,
                      torch.floor,
                      torch.round,
                      torch.trunc,
                      torch.frac,
                      torch.reciprocal,
                      torch.relu,
                      torch.sigmoid,
                      torch.tanh,
                      torch.nn.functional.gelu]
        prev_fallback = os.environ['PYTORCH_NVFUSER_DISABLE_FALLBACK']
        os.environ['PYTORCH_NVFUSER_DISABLE_FALLBACK'] = '0'
        for op, dtype in itertools.product(operations, dtypes):
            self._unary_type_test_helper(op, dtype)  # test special numbers
            self._unary_type_test_helper(op, dtype)  # test random data
        os.environ['PYTORCH_NVFUSER_DISABLE_FALLBACK'] = prev_fallback

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_unary_bitwise(self):
        def bit_not(x: torch.Tensor):
            return ~(x + 0)

        jitted = torch.jit.script(bit_not)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda").mul(5).to(torch.long)
        jit_o = bit_not(x)
        jit_o = bit_not(x)
        o = bit_not(x)
        self.assertEqual(o, jit_o)
        jitted.graph_for(x)  # Shows up in second instance, not first
        self.assertGraphContains(jitted.graph_for(x), FUSION_GUARD)

        def bool_not(x: torch.Tensor, y: torch.Tensor):
            return ~(x & y)

        jitted = torch.jit.script(bool_not)
        x = torch.rand(4, 8, 32, 32, dtype=torch.float, device="cuda").round().to(torch.bool)
        y = torch.rand(4, 8, 32, 32, dtype=torch.float, device="cuda").round().to(torch.bool)
        jit_o = bool_not(x, y)
        jit_o = bool_not(x, y)
        o = bool_not(x, y)
        self.assertEqual(o, jit_o)
        jitted.graph_for(x, y)  # Shows up in second instance, not first
        self.assertGraphContains(jitted.graph_for(x, y), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_binary_ops(self):
        operations = [torch.div,
                      torch.mul,
                      torch.atan2,
                      torch.max,
                      torch.min,
                      torch.pow,
                      torch.remainder,
                      torch.fmod,
                      torch.eq,
                      torch.ne,
                      torch.ge,
                      torch.gt,
                      torch.le,
                      torch.lt]
        for op in operations:
            self._binary_test_helper(op)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_binary_bitwise(self):
        def jit_or(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            return (x & y) | z

        def jit_xor(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            return (x & y) ^ z

        def jit_lshift(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            return (x & y) << z

        def jit_rshift(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            return (x & y) >> z

        for jit_func in [jit_or, jit_xor, jit_lshift, jit_rshift]:
            x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda").mul(5).to(torch.long)
            y = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda").mul(5).to(torch.long)
            z = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda").mul(2).to(torch.long)

            jitted = torch.jit.script(jit_func)
            jit_o = jitted(x, y, z)
            jit_o = jitted(x, y, z)
            o = jit_func(x, y, z)
            self.assertEqual(o, jit_o)
            self.assertGraphContains(jitted.graph_for(x, y, z), FUSION_GUARD)

        # We shouldn't need this redefinition of the function, but otherwise it won't recompile for a new type
        def jit_or(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            return x & y | z

        for jit_func in [jit_or, ]:
            x = torch.rand(4, 2, dtype=torch.float, device="cuda").round().to(torch.bool)
            y = torch.rand(4, 2, dtype=torch.float, device="cuda").round().to(torch.bool)
            z = torch.rand(4, 2, dtype=torch.float, device="cuda").round().to(torch.bool)

            jitted = torch.jit.script(jit_func)
            jit_o = jitted(x, y, z)
            jit_o = jitted(x, y, z)
            o = jit_func(x, y, z)
            self.assertEqual(o, jit_o)
            self.assertGraphContains(jitted.graph_for(x, y, z), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_type_as_op(self):
        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = torch.lt(x, z)
            o = o.type_as(y)
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 0.5)
        jit_o = t_jit(x, y, 0.5)
        o = t(x, y, 0.5)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, 0.5), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    # legacy fuser does not work for rand_like, see issue #34361
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires fusion optimization pass to be effective")
    def test_ternary_ops(self):
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        z = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        cond = torch.randint(0, 2, (4, 8, 32, 32)).to(dtype=torch.bool, device="cuda")

        def add(x: torch.Tensor, other: torch.Tensor, alpha: float):
            o = torch.relu(x)
            o = torch.add(o, other=other, alpha=alpha)
            return o
        add_jit = torch.jit.script(add)
        self._run_helper(add_jit, add, x, y, 2.0)

        def clamp0(x: torch.Tensor, f: float):
            o = torch.rand_like(x)
            o = o * torch.clamp(x, min=f)
            return o
        clamp0_jit = torch.jit.script(clamp0)
        self._run_helper(clamp0_jit, clamp0, x, 0.5)

        def clamp1(x: torch.Tensor, f: float, ff: float):
            o = torch.rand_like(x)
            o = o * torch.clamp(x, min=f, max=ff)
            return o
        clamp1_jit = torch.jit.script(clamp1)
        self._run_helper(clamp1_jit, clamp1, x, -0.2, 0.7)

        def threshold(x: torch.Tensor, th: float, val: float):
            o = torch.rand_like(x)
            o = x * torch.threshold(o, th, val)
            return o
        threshold_jit = torch.jit.script(threshold)
        self._run_helper(threshold_jit, threshold, x, 0.2, 0.9)

        def where(x: torch.Tensor, y: torch.Tensor, cond: torch.Tensor):
            o = torch.rand_like(x)
            o = o * torch.where(cond, x, y)
            return o
        where_jit = torch.jit.script(where)
        self._run_helper(where_jit, where, x, y, cond)

        def lerp(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = torch.rand_like(x)
            o = o * torch.lerp(x, y, z)
            return o
        lerp_jit = torch.jit.script(lerp)
        self._run_helper(lerp_jit, lerp, x, y, z)

        def lerp_scale(x: torch.Tensor, y: torch.Tensor, z: float):
            o = torch.rand_like(x)
            o = o * torch.lerp(x, y, z)
            return o
        lerp_scale_jit = torch.jit.script(lerp_scale)
        self._run_helper(lerp_scale_jit, lerp_scale, x, y, 0.5)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires profiling node to run cuda fuser")
    def test_addcmul_ops(self):
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        z = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")

        def addcmul(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, value: float):
            o = torch.add(x, 0.5)
            o = torch.addcmul(o, y, z, value=value)
            return o
        addcmul_jit = torch.jit.script(addcmul)
        self._run_helper(addcmul_jit, addcmul, x, y, z, 2.0)

        def addcmul_no_alpha(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = torch.add(x, 0.5)
            o = torch.addcmul(o, y, z)
            return o
        addcmul_no_alpha_jit = torch.jit.script(addcmul_no_alpha)
        self._run_helper(addcmul_no_alpha_jit, addcmul_no_alpha, x, y, z)

        def addcmul_const_alpha(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = torch.add(x, 0.5)
            o = torch.addcmul(o, y, z, value=0.75)
            return o
        addcmul_const_alpha_jit = torch.jit.script(addcmul_const_alpha)
        self._run_helper(addcmul_const_alpha_jit, addcmul_const_alpha, x, y, z)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_dynamic_size(self):
        old_guard = torch._C._jit_set_nvfuser_guard_mode(True)
        torch._C._jit_set_bailout_depth(20)

        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        subgraph = self._getSubgraphInFusion(t_jit.graph_for(x, y, 2.0))
        self.assertGraphContainsExactly(subgraph, 'aten::add', 2, consider_subgraphs=False)

        # this test is not ideal, as we rely on the bailout to test it and we
        # don't know a way to verify the bailout graph to validate the proper
        # fusion.
        x = torch.randn(8, 32, 16, 8, dtype=torch.float, device="cuda")
        y = torch.randn(16, 8, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, 2.0), FUSION_GUARD)
        x = torch.randn(8, 17, 8, dtype=torch.float, device="cuda")
        y = torch.randn(8, 17, 1, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, 2.0), FUSION_GUARD)
        torch._C._jit_set_nvfuser_guard_mode(old_guard)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_random_topo(self):
        os.environ["PYTORCH_NVFUSER_DISABLE_FALLBACK"] = "1"
        self.assertTrue(runDefaultTestWithSeed(28449))

    def _compare(self, desc, inp1, inp2, error):
        a = inp1.clone().detach().cpu().numpy()
        b = inp2.clone().detach().cpu().numpy()
        close = np.allclose(a, b, error, error)
        if not close:
            print(desc, close)
            z = a - b
            index = (np.abs(z) >= error + error * np.abs(b)).nonzero()
            print("dif    : ", z[index])
            print("inp1   : ", a[index])
            print("inp2   : ", b[index])
        return close

    # Permutation helper that applies binary operation between two tensors:
    #   1. applies separate permutation `perm0` & `perm1` to two inputs
    #   2. reduce dimension `broadcast_axis` of operand two to size 1
    # The purpose of this test is to ensure permutation works well in
    # complicated cases with arbitrary stride order and broadcasting dimensions
    def _permutation_helper(self, sizes, broadcast_axis, dtype, device, perm0, perm1):
        def t(x: torch.Tensor, y: torch.Tensor):
            o = torch.add(x, y)
            o = torch.relu(o)
            return o

        x = torch.randn([sizes[i] for i in perm0], dtype=dtype, device=device).permute(
            [perm0.index(i) for i in range(len(sizes))])
        if broadcast_axis >= 0:
            sizes[broadcast_axis] = 1
        y = torch.randn([sizes[i] for i in perm1], dtype=dtype, device=device).permute(
            [perm1.index(i) for i in range(len(sizes))])
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y), FUSION_GUARD)

    # end-2-end test of permutation & contiguity handling in integration.
    # we are testing inputs with all combination of permutation order, just to
    # ensure that integration would be able to generate functionally correct
    # kernels
    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_binary_ops_permutation(self):
        # note that num_dim is exclusive from len(x), so we are not reducing
        # to single element (codegen limitation at this moment)
        x = [7, 8, 12]
        b_axes = range(-1, len(x))
        for b_axis in b_axes:
            for perm0 in itertools.permutations(range(len(x))):
                for perm1 in itertools.permutations(range(len(x))):
                    x = [7, 8, 12]
                    self._permutation_helper(x, b_axis, torch.float32, "cuda", perm0, perm1)

    def _reduction_helper(self, sizes, reduction_axis, dtype, device, perm0, perm1, keepdim=False):
        class MyReduction(torch.nn.Module):
            __constants__ = ['reduction_axis', 'keepdim']

            def __init__(self):
                super(MyReduction, self).__init__()
                self.reduction_axis = reduction_axis
                self.keepdim = keepdim

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                o = torch.add(x, y)
                o = torch.sum(o, dim=self.reduction_axis, keepdim=self.keepdim)
                return o

        t = MyReduction()

        x = torch.randn([sizes[i] for i in perm0], dtype=dtype, device=device).permute(
            [perm0.index(i) for i in range(len(sizes))])
        y = torch.randn([sizes[i] for i in perm1], dtype=dtype, device=device).permute(
            [perm1.index(i) for i in range(len(sizes))])
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)
        self.assertEqual(o.dtype, jit_o.dtype)
        # numerical issues here due to our scheduling.
        # can't use `self.assertEqual(o, jit_o)`
        self.assertTrue(self._compare("comparing output failed", o, jit_o, 1e-4))
        self.assertGraphContains(t_jit.graph_for(x, y), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_reduction(self):
        for x in ([7, 8, 12], [12, 8, 7, 9, 15], [128, 16, 8, 32]):
            # note that num_dim is exclusive from len(x), so we are not reducing
            # to single element (codegen limitation at this moment)
            for num_reduce_dim in range(1, len(x)):
                for axes in itertools.combinations(range(len(x)), num_reduce_dim):
                    for keepdim in (True, False):
                        perm0 = range(len(x))
                        perm1 = range(len(x))
                        self._reduction_helper(x, axes, torch.float32, "cuda", perm0, perm1, keepdim)

    def _layer_norm_helper(self, shape, norm_shape, dtype, device, error):
        class MyLayerNorm(torch.nn.Module):
            __constants__ = ['norm_shape']

            def __init__(self):
                super(MyLayerNorm, self).__init__()
                self.norm_shape = norm_shape

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                o = torch.add(x, y)
                o = torch.nn.functional.layer_norm(o, self.norm_shape)
                return o

        t = MyLayerNorm()

        x = torch.randn(shape, dtype=dtype, device=device)
        y = torch.randn(shape, dtype=dtype, device=device)
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)
        self.assertEqual(o.dtype, jit_o.dtype)
        # numerical issues here due to our scheduling.
        # can't use `self.assertEqual(o, jit_o)`
        self.assertTrue(self._compare("comparing output failed", o, jit_o, error))
        self.assertGraphContains(t_jit.graph_for(x, y), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_layer_norm(self):
        dims = 4
        rnds = 3
        for idx in range(rnds):
            for offset in range(1, dims):
                input_shape = [random.randint(30, 100) for idx in range(dims)]
                norm_shape = [input_shape[idx] for idx in range(dims - offset, dims)]
                self._layer_norm_helper(input_shape, norm_shape, torch.float32, "cuda", 1e-4)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_layer_norm_half(self):
        dims = 4
        rnds = 3
        for idx in range(rnds):
            for offset in range(1, dims):
                input_shape = [random.randint(30, 100) for idx in range(dims)]
                norm_shape = [input_shape[idx] for idx in range(dims - offset, dims)]
                self._layer_norm_helper(input_shape, norm_shape, torch.float16, "cuda", 5e-3)

    def _batch_norm_helper(self, shape, dtype, device, error):
        class MyBatchNorm(torch.nn.Module):
            def __init__(self):
                super(MyBatchNorm, self).__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor, r_mean: torch.Tensor, r_var: torch.Tensor):
                o = torch.add(x, y)
                o = torch.nn.functional.batch_norm(o, r_mean, r_var, training=True)
                return o

        t = MyBatchNorm()

        x = torch.randn(shape, dtype=dtype, device=device)
        y = torch.randn(shape, dtype=dtype, device=device)
        running_mean = torch.randn(shape[1], dtype=torch.float32, device=device)
        running_var = torch.randn(shape[1], dtype=torch.float32, device=device)
        t_jit = torch.jit.script(t)

        eager_running_mean = running_mean.clone()
        eager_running_var = running_var.clone()
        jit_running_mean = running_mean.clone()
        jit_running_var = running_var.clone()

        jit_o = t_jit(x, y, running_mean.clone(), running_var.clone())
        jit_o = t_jit(x, y, jit_running_mean, jit_running_var)
        o = t(x, y, eager_running_mean, eager_running_var)
        self.assertEqual(o.dtype, jit_o.dtype)
        # numerical issues here due to our scheduling.
        # can't use `self.assertEqual(o, jit_o)`
        self.assertTrue(self._compare("comparing output failed", o, jit_o, error))
        # TODO: enable checks when we support in-place updates for batch_norm tensors
        # self.assertTrue(self._compare("comparing output failed", eager_running_mean, jit_running_mean, error))
        # self.assertTrue(self._compare("comparing output failed", eager_running_var, jit_running_var, error))
        self.assertGraphContains(t_jit.graph_for(x, y, running_mean, running_var), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_batch_norm(self):
        output_elements = 10000
        channel_sizes = [67, 457, 1024, 4096]

        for dims in range(3, 6):
            output_size = int(pow(output_elements, 1. / (dims - 1)))
            for C in channel_sizes:
                x = [output_size for idx in range(dims)]
                x[1] = C
                self._batch_norm_helper(x, torch.float32, "cuda", 1e-4)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_batch_norm_half(self):
        output_elements = 10000
        channel_sizes = [67, 457, 1024, 4096]

        for dims in range(3, 6):
            output_size = int(pow(output_elements, 1. / (dims - 1)))
            for C in channel_sizes:
                x = [output_size for idx in range(dims)]
                x[1] = C
                self._batch_norm_helper(x, torch.float16, "cuda", 5e-3)

    def _softmax_helper(self, shape, reduction_axis, dtype, device, error):
        class MySoftmax(torch.nn.Module):
            __constants__ = ['reduction_axis']

            def __init__(self):
                super(MySoftmax, self).__init__()
                self.reduction_axis = reduction_axis

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                o = torch.add(x, y)
                o = torch.nn.functional.softmax(o, dim=self.reduction_axis)
                return o

        t = MySoftmax()

        x = torch.randn(shape, dtype=dtype, device=device)
        y = torch.randn(shape, dtype=dtype, device=device)
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)
        self.assertEqual(o.dtype, jit_o.dtype)
        # numerical issues here due to our scheduling.
        # can't use `self.assertEqual(o, jit_o)`
        self.assertTrue(self._compare("comparing output failed", o, jit_o, error))
        self.assertGraphContains(t_jit.graph_for(x, y), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_softmax(self):
        output_size = 10000
        dims = 4
        output_size = int(pow(output_size, 1. / dims))
        reduction_sizes = [67, 256, 1024, 4096]

        for reduction_dim in range(dims):
            for reduction_size in reduction_sizes:
                x = [output_size for idx in range(dims)]
                x[reduction_dim] = reduction_size
                self._softmax_helper(x, reduction_dim, torch.float32, "cuda", 1e-4)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_softmax_half(self):
        output_size = 10000
        dims = 4
        output_size = int(pow(output_size, 1. / dims))
        reduction_sizes = [67, 256, 1024, 4096]

        for reduction_dim in range(dims):
            for reduction_size in reduction_sizes:
                x = [output_size for idx in range(dims)]
                x[reduction_dim] = reduction_size
                self._softmax_helper(x, reduction_dim, torch.float16, "cuda", 5e-3)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_reduction_permutation(self):
        x = [7, 8, 12]
        # note that num_dim is exclusive from len(x), so we are not reducing
        # to single element (codegen limitation at this moment)
        for num_reduce_dim in range(1, len(x)):
            for axes in itertools.combinations(range(len(x)), num_reduce_dim):
                for perm0 in itertools.permutations(range(len(x))):
                    for perm1 in itertools.permutations(range(len(x))):
                        self._reduction_helper(x, axes, torch.float32, "cuda", perm0, perm1)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_reduction_multiple_output(self):
        old_guard = torch._C._jit_set_nvfuser_guard_mode(True)
        torch._C._jit_set_bailout_depth(20)

        def t(x: torch.Tensor, y: torch.Tensor, scale: float, z: torch.Tensor):
            o = torch.mul(x, y)
            o = torch.mul(o, scale)
            out1 = torch.mul(o, z)
            out2 = torch.sum(out1, dim=[2])
            return out1, out2

        t_jit = torch.jit.script(t)
        x = torch.randn(8, 4, 10, 16, dtype=torch.float, device="cuda")
        y = torch.randn(8, 4, 10, 16, dtype=torch.float, device="cuda")
        z = torch.randn(8, 4, 10, 16, dtype=torch.float, device="cuda")
        scale = 0.5
        jit_o = t_jit(x, y, scale, z)
        jit_o = t_jit(x, y, scale, z)
        o = t(x, y, scale, z)
        for oo, jit_oo in zip(o, jit_o):
            self.assertEqual(oo.dtype, jit_oo.dtype)
            self.assertEqual(oo, jit_oo)
        self.assertGraphContains(t_jit.graph_for(x, y, scale, z), FUSION_GUARD)

        x = x.to(memory_format=torch.channels_last)
        y = y.to(memory_format=torch.channels_last)
        z = z.to(memory_format=torch.channels_last)
        jit_o = t_jit(x, y, scale, z)
        jit_o = t_jit(x, y, scale, z)
        o = t(x, y, scale, z)
        for oo, jit_oo in zip(o, jit_o):
            self.assertEqual(oo.dtype, jit_oo.dtype)
            self.assertEqual(oo, jit_oo)
        self.assertGraphContains(t_jit.graph_for(x, y, scale, z), FUSION_GUARD)
        torch._C._jit_set_nvfuser_guard_mode(old_guard)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_reduction_dtype(self):
        def t(x: torch.Tensor):
            o = torch.mul(x, 1.0)
            o = torch.sum(o, dim=[2], dtype=torch.float32)
            return o
        t_jit = torch.jit.script(t)

        x = torch.randn(8, 4, 16, dtype=torch.float, device="cuda")
        jit_o = t_jit(x)
        jit_o = t_jit(x)
        o = t(x)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertTrue(self._compare("comparing output failed", o, jit_o, 1e-4))
        self.assertGraphContains(t_jit.graph_for(x), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_reduction_half(self):
        def t(x: torch.Tensor):
            o = torch.mul(x, 1.0)
            o = torch.sum(o, dim=[2])
            return o

        t_jit = torch.jit.script(t)
        x = torch.randn(8, 4, 16, dtype=torch.float16, device="cuda")
        jit_o = t_jit(x)
        jit_o = t_jit(x)
        o = t(x)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertTrue(self._compare("comparing output failed", o, jit_o, 1e-4))
        self.assertGraphContains(t_jit.graph_for(x), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_pw_single_reduction_partition(self):
        sizes = [8, 8, 8]
        dtype = torch.float
        device = "cuda"
        x = torch.randn(sizes, dtype=dtype, device=device)
        y = torch.randn(sizes, dtype=dtype, device=device)
        z = torch.randn(sizes, dtype=dtype, device=device)

        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = torch.add(x, y)
            o = torch.sum(o, dim=[0])
            o = torch.add(o, z)
            return o
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, z), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_sum_to_one(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn([4, 5, 6], dtype=dtype, device=device)

        def t(x: torch.Tensor):
            o = torch.add(x, 0)
            o = torch.sum(o, dim=[0, 1, 2])
            return o
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x)
        jit_o = t_jit(x)
        o = t(x)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_single_reduction_broadcast(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn([7, 4, 8], dtype=dtype, device=device)
        y = torch.randn([4, 8], dtype=dtype, device=device)
        z = torch.randn([1, 4, 8], dtype=dtype, device=device)

        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = torch.add(x, y)
            o = torch.add(o, z)
            o = torch.sum(o, dim=[0])
            return o
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, z), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_trivial_reduction(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn([1, 4, 8], dtype=dtype, device=device)

        def t(x: torch.Tensor):
            o = torch.add(x, 0)
            o = torch.sum(o, dim=[0])
            o = torch.sum(o, dim=[0])
            return o
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x)
        jit_o = t_jit(x)
        o = t(x)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x), FUSION_GUARD)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_profiling_node(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn(4, 8, 8, 8, dtype=dtype, device=device)

        def repro(x: torch.Tensor, alpha: float):
            o = torch.rand_like(x)
            o = torch.add(o, alpha)
            return o
        repro_jit = torch.jit.script(repro)
        self._run_helper(repro_jit, repro, x, 0.6)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_reduction_sizes_op(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn(2, 3, 4, 5, dtype=dtype, device=device)
        y = torch.randn(2, 3, 4, 5, dtype=dtype, device=device)

        def t(x: torch.Tensor, y: torch.Tensor):
            o = x + y
            o = torch.relu(o)
            o = o.sum((1, 3))
            return o.size()
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)
        self.assertEqual(o, jit_o)
        # since the output value is not used at all, the fusion operator should
        # have been optimized away
        self.assertGraphContainsExactly(t_jit.graph_for(x, y), FUSION_GUARD, 0)


class TestPassManagerCudaFuser(JitTestCase):

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_context_manager_test(self):
        x = torch.randn(4, 8, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, dtype=torch.float, device="cuda")
        with torch.jit.fuser('fuser2'):
            with torch.jit.fuser('fuser2'):

                def t1(x, y):
                    o = x + y
                    o = o + 2.0
                    return o
                t_jit = torch.jit.script(t1)
                t_jit(x, y)
                t_jit(x, y)
                self.assertGraphContains(t_jit.graph_for(x, y), FUSION_GUARD)

            def t2(x, y):
                o = x + y
                o = o + 3.0
                return o
            t_jit_2 = torch.jit.script(t2)
            t_jit_2(x, y)
            t_jit_2(x, y)
            self.assertGraphContains(t_jit_2.graph_for(x, y), FUSION_GUARD)

        def t3(x, y):
            o = x + y
            o = o + 4.0
            return o
        t_jit_3 = torch.jit.script(t3)
        t_jit_3(x, y)
        t_jit_3(x, y)
        self.assertGraphContainsExactly(t_jit_3.graph_for(x, y), FUSION_GUARD, 0)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_register_fuser(self):
        self.assertFalse(torch._C._jit_set_nvfuser_enabled(True))
        self.assertTrue(torch._C._jit_nvfuser_enabled())
        self.assertTrue(torch._C._jit_set_nvfuser_enabled(True))
        self.assertTrue(torch._C._jit_nvfuser_enabled())
        self.assertTrue(torch._C._jit_set_nvfuser_enabled(False))
        self.assertFalse(torch._C._jit_nvfuser_enabled())


if __name__ == '__main__':
    run_tests()
