# Owner(s): ["module: inductor"]
import copy
import os
import tempfile
import unittest

import torch
from torch._precompile import PrecompileError
from torch.testing._internal.common_utils import run_tests, TestCase


# A module-level (global) model + a function referencing it, to exercise the
# constant-tensor guard against a baked global.
_GLOBAL_TENSOR = torch.randn(3)


class TestPrecompile(TestCase):
    def test_plain_function(self):
        def f(x, y):
            return (x @ y).sin(), x + y

        a, b = torch.randn(4, 4), torch.randn(4, 4)
        code, cache = torch.precompile(f, a, b)
        self.assertIsInstance(code, str)
        self.assertIsInstance(cache, bytes)

        f_c = torch.precompile.load(code, cache)
        out = f_c(a, b)
        ref = f(a, b)
        self.assertEqual(out[0], ref[0])
        self.assertEqual(out[1], ref[1])

    def test_module_params_and_buffers_are_lifted(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.register_buffer("b2", torch.randn(3))

            def forward(self, x):
                return torch.relu(self.lin(x)) + self.b2

        m = M().eval()
        x = torch.randn(5, 4)
        code, cache = torch.precompile(lambda model, x: model(x), m, x)
        f_c = torch.precompile.load(code, cache)
        self.assertEqual(f_c(x), m(x))

    def test_constant_tensor_is_rejected(self):
        captured = torch.randn(3)
        with self.assertRaisesRegex(PrecompileError, "hard-coded"):
            torch.precompile(lambda x: x + captured, torch.randn(3))

    def test_global_tensor_rejected_unlike_make_fx(self):
        # Vanilla make_fx silently bakes a referenced global tensor into the
        # GraphModule as a get_attr constant; precompile must instead error.
        from torch.fx.experimental.proxy_tensor import make_fx

        def f(x):
            return x + _GLOBAL_TENSOR

        gm = make_fx(f)(torch.randn(3))
        baked = [
            n.target
            for n in gm.graph.nodes
            if n.op == "get_attr"
            and isinstance(getattr(gm, n.target, None), torch.Tensor)
        ]
        self.assertTrue(baked, "expected vanilla make_fx to bake a tensor constant")

        with self.assertRaisesRegex(PrecompileError, "hard-coded"):
            torch.precompile(f, torch.randn(3))

    def test_unregistered_module_tensor_attr_is_rejected(self):
        # A plain tensor attribute (not a registered parameter/buffer) is not
        # lifted, so referencing it would bake it in -- this must error.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4))
                self.scale = torch.randn(4)  # plain attr, NOT a buffer/parameter

            def forward(self, x):
                return (x @ self.weight) * self.scale

        m = M().eval()
        with self.assertRaisesRegex(PrecompileError, "hard-coded"):
            torch.precompile(lambda model, x: model(x), m, torch.randn(2, 4))

    def test_export_and_reload_roundtrip(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.register_buffer("b2", torch.randn(3))

            def forward(self, x):
                return torch.relu(self.lin(x)) + self.b2

        m = M().eval()
        x = torch.randn(5, 4)
        code, cache = torch.precompile(lambda model, x: model(x), m, x)

        self.assertIn("Inductor output code", code)
        self.assertIn("def forward(", code)
        self.assertIn("PARAM_NAMES = ['lin.weight', 'lin.bias']", code)

        f_c = torch.precompile.load(code, cache)
        self.assertEqual(f_c(x), m(x))

    def test_self_contained_exec(self):
        # Executing the python_code string standalone (with the cache written to
        # its WEIGHTS_PATH) defines a runnable forward().
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.precompile(lambda model, x: model(x), m, x)

        with tempfile.TemporaryDirectory() as d:
            cache_path = os.path.join(d, "cache.bin")
            with open(cache_path, "wb") as fh:
                fh.write(cache)
            ns = {"__name__": "_artifact"}
            exec(compile(code, "<artifact>", "exec"), ns)
            ns["WEIGHTS_PATH"] = cache_path
            self.assertEqual(ns["forward"](x), m(x))

    def test_cache_primes_inductor_on_reload(self):
        # Reloading in a fresh inductor cache dir primes it and hits FxGraphCache
        # (no re-lowering) -- the kernel caching the cache provides.
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_cache

        m = torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
        ).eval()
        x = torch.randn(3, 8)
        code, cache = torch.precompile(lambda model, x: model(x), m, x)

        with fresh_cache():
            counters.clear()
            f_c = torch.precompile.load(code, cache)
            self.assertEqual(f_c(x), m(x))
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA for Triton autotuning")
    def test_cache_bundles_autotune_artifacts(self):
        from torch._inductor.utils import fresh_cache

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(512, 512)
                self.l2 = torch.nn.Linear(512, 512)

            def forward(self, x):
                return torch.softmax(self.l2(torch.relu(self.l1(x))), dim=-1)

        m = M().cuda().eval()
        x = torch.randn(128, 512, device="cuda")
        code, cache = torch.precompile(lambda model, x: model(x), m, x)
        with fresh_cache():
            f_c = torch.precompile.load(code, cache)
            self.assertEqual(f_c(x), m(x))

    def test_dtensor_subclass(self):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo not available")

        from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29555")
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            mesh = DeviceMesh("cpu", list(range(1)))
            m = torch.nn.Linear(4, 3).eval()
            for name, p in list(m.named_parameters()):
                setattr(
                    m,
                    name,
                    torch.nn.Parameter(
                        distribute_tensor(p.detach(), mesh, [Replicate()])
                    ),
                )
            x = distribute_tensor(torch.randn(5, 4), mesh, [Replicate()])
            ref = m(x)

            code, cache = torch.precompile(lambda model, x: model(x), m, x)
            f_c = torch.precompile.load(code, cache)
            out = f_c(x)
            self.assertEqual(out.to_local(), ref.to_local())
        finally:
            dist.destroy_process_group()

    def test_training_backward_harvest_matches_eager(self):
        # A training step that calls loss.backward(): precompile harvests the
        # parameter grads as outputs; they match eager autograd.
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        )
        loss_fn = torch.nn.MSELoss()
        x = torch.randn(5, 4)
        target = torch.randn(5, 3)

        ref = copy.deepcopy(model)
        loss_fn(ref(x), target).backward()
        ref_grads = [p.grad.clone() for p in ref.parameters()]

        def train_step(model, x, target):
            loss_fn(model(x), target).backward()

        code, cache = torch.precompile(train_step, model, x, target)
        f_c = torch.precompile.load(code, cache)

        grads = f_c(x, target)  # result is None -> output is the grads
        for g, rg in zip(grads, ref_grads):
            self.assertEqual(g, rg)

        # A multi-step loop (apply harvested grads via an optimizer) reduces loss.
        opt = torch.optim.SGD(f_c.parameters(), lr=0.1)

        def loss_of(params, x, target):
            return loss_fn(
                torch.nn.functional.linear(
                    torch.relu(torch.nn.functional.linear(x, params[0], params[1])),
                    params[2],
                    params[3],
                ),
                target,
            )

        losses = []
        for _ in range(5):
            grads = f_c(x, target)
            for p, g in zip(f_c.parameters(), grads):
                p.grad = g
            losses.append(loss_of(f_c.parameters(), x, target).item())
            opt.step()
        self.assertLess(losses[-1], losses[0])

if __name__ == "__main__":
    run_tests()
