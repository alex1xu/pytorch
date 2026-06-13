"""make_fx-based ahead-of-time precompilation.

``python_code, cache = torch.precompile(fn, model, *example_inputs)`` traces the
computation ``fn`` (e.g. ``lambda model, x: model(x)``) with ``make_fx`` in a
principled way: the module's parameters and buffers are lifted to explicit graph
inputs (via functional reparametrization) so that no live tensor is ever baked
into the graph as a constant. Any tensor that is neither a graph input
(parameter/buffer/user input) nor an intermediate is rejected, since silently
hard-coding such a tensor into the graph would be a correctness footgun.

The captured graph is lowered through ``torch._inductor.standalone_compile``,
which runs the full AOTAutograd + Inductor pipeline (runtime wrappers plus
generated output code). ``precompile`` returns a self-contained, executable
Python source string and a binary cache (parameter values + the real
Inductor/AOTAutograd artifact); reload a runnable with
``torch.precompile.load(python_code, cache)``.
"""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Callable
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless


__all__ = ["precompile", "PrecompileError"]

# Default WEIGHTS_PATH baked into the python artifact for standalone exec; the
# load() path supplies the cache directly and does not rely on it.
_DEFAULT_WEIGHTS_PATH = "precompile_cache.bin"


class PrecompileError(RuntimeError):
    """Raised when precompile tracing would bake a tensor into the graph."""


def _check_no_constant_tensors(gm: torch.fx.GraphModule) -> None:
    """Reject graphs that hard-code a tensor as a ``get_attr`` constant.

    Every legitimate tensor in a non-strict capture is a placeholder (a lifted
    parameter/buffer or user input) or the result of a ``call_function`` node.
    A ``get_attr`` pointing at a tensor therefore means some tensor was closed
    over and would be baked into the graph, which we forbid.
    """
    offending = []
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        attr = gm
        for part in node.target.split("."):
            attr = getattr(attr, part, None)
        if isinstance(attr, torch.Tensor):
            offending.append((node.target, tuple(attr.shape), str(attr.dtype)))
    if offending:
        raise PrecompileError(
            "precompile traced a tensor that is neither a graph input "
            "(module parameter/buffer or user input) nor an intermediate. Such "
            "tensors would be hard-coded into the graph. Offending constants "
            f"(target, shape, dtype): {offending}. Pass these tensors as inputs, "
            "or register them as parameters/buffers on the module."
        )


def _capture(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    decompositions: dict | None = None,
) -> _Capture:
    """Trace the computation ``fn(*args)`` to an ATen graph.

    ``fn`` is the whole computation, e.g. ``lambda model, x: model(x)`` or a
    training step ``lambda model, x, t: loss_fn(model(x), t).backward()``. Among
    ``args``, the ``nn.Module`` arguments have their parameters/buffers lifted to
    explicit graph inputs (via reparametrization, so nothing is baked); the
    remaining arguments are the runtime inputs. Whatever ``fn`` returns becomes a
    graph output, and if ``fn`` ran a backward, the resulting parameter
    gradients (read off ``param.grad``) are harvested and appended as outputs.
    """
    import contextlib

    args = tuple(args)
    module_positions = [i for i, a in enumerate(args) if isinstance(a, torch.nn.Module)]
    module_pos_set = set(module_positions)
    mods = [args[i] for i in module_positions]
    user_inputs = tuple(a for i, a in enumerate(args) if i not in module_pos_set)

    param_entries: list[tuple[int, str, Any]] = []
    buffer_entries: list[tuple[int, str, Any]] = []
    for mi, m in enumerate(mods):
        for n, p in m.named_parameters(remove_duplicate=False):
            param_entries.append((mi, n, p))
        for n, b in m.named_buffers(remove_duplicate=False):
            buffer_entries.append((mi, n, b))
    pb_entries = param_entries + buffer_entries
    pb_keys = [(mi, n) for mi, n, _ in pb_entries]
    pb_flat = [v for _, _, v in pb_entries]
    num_pb = len(pb_flat)
    num_params = len(param_entries)

    multi = len(mods) > 1

    def _name(mi: int, n: str) -> str:
        return f"m{mi}.{n}" if multi else n

    param_names = [_name(mi, n) for mi, n, _ in param_entries]
    buffer_names = [_name(mi, n) for mi, n, _ in buffer_entries]

    user_flat, in_spec = pytree.tree_flatten(user_inputs)
    flat_args = [*pb_flat, *user_flat]

    out_spec_holder: dict[str, pytree.TreeSpec] = {}

    def flat_fn(flat: list[Any]) -> list[Any]:
        pb = flat[:num_pb]
        runtime_inputs = pytree.tree_unflatten(flat[num_pb:], in_spec)
        with contextlib.ExitStack() as stack:
            for mi, m in enumerate(mods):
                reparam = {n: v for (k, n), v in zip(pb_keys, pb) if k == mi}
                stack.enter_context(
                    stateless._reparametrize_module(m, reparam, tie_weights=True)
                )
            # Reconstruct fn's full positional args: reparametrized modules at
            # their original positions, runtime inputs at theirs.
            full: list[Any] = []
            ui = 0
            for i in range(len(args)):
                if i in module_pos_set:
                    full.append(args[i])
                else:
                    full.append(runtime_inputs[ui])
                    ui += 1
            result = fn(*full)
            # Harvest parameter gradients produced by any backward in fn.
            param_proxies = pb[:num_params]
            harvested = [p.grad for p in param_proxies]

        if any(g is not None for g in harvested):
            grads = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(harvested, param_proxies)
            ]
            out: Any = grads if result is None else (result, grads)
        else:
            out = result
        out_flat, out_spec = pytree.tree_flatten(out)
        out_spec_holder["spec"] = out_spec
        return out_flat

    # Trace with grad enabled so any backward in ``fn`` is built as graph ops; the
    # forward graph is the same as under no_grad.
    with torch.enable_grad():
        gm = make_fx(flat_fn, decomposition_table=decompositions)(flat_args)
    _check_no_constant_tensors(gm)

    return _Capture(
        gm=gm,
        flat_args=flat_args,
        param_names=param_names,
        buffer_names=buffer_names,
        param_buffer_flat=pb_flat,
        num_params_buffers=num_pb,
        in_spec=in_spec,
        out_spec=out_spec_holder["spec"],
    )


class _Capture:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        flat_args: list[Any],
        param_names: list[str],
        buffer_names: list[str],
        param_buffer_flat: list[Any],
        num_params_buffers: int,
        in_spec: pytree.TreeSpec,
        out_spec: pytree.TreeSpec,
    ) -> None:
        self.gm = gm
        self.flat_args = flat_args
        self.param_names = param_names
        self.buffer_names = buffer_names
        self.param_buffer_flat = param_buffer_flat
        self.num_params_buffers = num_params_buffers
        self.in_spec = in_spec
        self.out_spec = out_spec


# AOTAutograd codegens its runtime wrappers as Python source. We observe them via
# _codegen_exec.capture_generated_wrappers (a supported hook) during compilation,
# then re-emit and re-compose them here so the exported file runs the *same*
# prologue/epilogue, subclass unwrap/rewrap, dedup, synthetic-base, RNG and
# effect-token handling that the JIT path uses. Helpers are built first (referenced
# by the orchestration via globals); the chain is composed inner -> outer.
_WRAPPER_HELPER_NAMES = ["mutation_epilogue", "output_alias_wrapper"]
_WRAPPER_CHAIN_NAMES = [
    "effect_tokens_wrapper",
    "subclass_wrapper",
    "functionalized_rng_wrapper",
    "runtime_wrapper_orchestration",
    "synthetic_base_wrapper",
    "dedup_wrapper",
]


_GENERATED_HEADER = """\
# Generated by torch.precompile -- do not edit.
#
# This is a SELF-CONTAINED, EXECUTABLE end-to-end artifact. You can run it
# directly:
#
#     python this_file.py                       # reports it is ready
#
# or import / exec it and call forward() with your own inputs:
#
#     ns = {}
#     exec(open("this_file.py").read(), ns)
#     out = ns["forward"](my_input)             # returns the model output
#
# It contains, in order:
#   1. The Inductor-generated output code (the actual compiled kernels and the
#      ``call(args)`` entry point). The kernels are (re)compiled from the source
#      embedded below on first use, so no external kernel cache is required.
#   2. Calling-convention metadata: the parameter/buffer names lifted to graph
#      inputs, and the pytree specs for user inputs and outputs.
#   3. The AOTAutograd runtime wrappers, captured verbatim from the codegen that
#      runs during compilation (prologue/epilogue, input-mutation and output-
#      alias handling, tensor-subclass unwrap/rewrap such as DTensor, dedup,
#      synthetic-base, functionalized RNG, effect tokens). These are recomposed
#      inner -> outer to form the boxed runtime function, which ``forward`` drives.
#
# Parameter/buffer VALUES and the residual wrapper globals (subclass metadata,
# etc.) are not embedded here; they come from the ``cache`` returned by
# precompile. torch.precompile.load(python_code, cache) supplies it; for
# standalone exec, write the cache bytes to WEIGHTS_PATH (see below).
"""


def _py_str_literal(s: str) -> str:
    """Emit ``s`` as a readable triple-quoted literal when safe, else repr."""
    if '"""' not in s and "\\" not in s:
        return 'r"""\n' + s + '\n"""'
    return repr(s)


def _build_metadata_section(
    compiled: NonStrictCompiled, weights_path: str
) -> list[str]:
    assert compiled._out_spec is not None
    out_spec_str = pytree.treespec_dumps(compiled._out_spec)
    parts = [
        "# " + "=" * 70,
        "# 2. Calling-convention metadata",
        "# " + "=" * 70,
        "import os as _os",
        "import torch as _torch",
        "import torch.utils._pytree as _pytree",
        "import importlib as _importlib",
        "import contextlib as _contextlib",
        "",
        f"PARAM_NAMES = {compiled._param_names!r}",
        f"BUFFER_NAMES = {compiled._buffer_names!r}",
        f"NUM_PARAMS_BUFFERS = {compiled._num_params_buffers}",
        f"OUT_SPEC = {out_spec_str!r}",
        "",
        "# Companion cache (the ``cache`` bytes returned by precompile, written to",
        "# a file): parameter/buffer values, residual wrapper globals, and the",
        "# inductor/AOTAutograd cache artifact. Override with the env var below or",
        "# by setting this module global before the first forward() call.",
        f'WEIGHTS_PATH = _os.environ.get("TORCH_PRECOMPILE_CACHE", {weights_path!r})',
        "",
    ]
    return parts


def _build_wrapper_section(compiled: NonStrictCompiled) -> list[str]:
    parts = [
        "# " + "=" * 70,
        "# 3. AOTAutograd runtime wrappers (captured from codegen)",
        "# " + "=" * 70,
    ]
    for rec in compiled._wrapper_records:
        parts.append(f"# ---- generated wrapper: {rec['name']} ----")
        parts.append(f"_SRC_{rec['name']} = {_py_str_literal(rec['source'])}")
        parts.append("")
    record_entries = []
    for rec in compiled._wrapper_records:
        record_entries.append(
            "    {"
            f'"name": {rec["name"]!r}, '
            f'"fn_name": {rec["fn_name"]!r}, '
            f'"plan": {rec["plan"]!r}, '
            f'"source": _SRC_{rec["name"]},'
            "}"
        )
    parts.append("_RECORDS = [\n" + ",\n".join(record_entries) + "\n]")
    parts.append("")
    return parts


def _build_python_source(
    compiled: NonStrictCompiled,
    inductor_chunks: list[str],
    weights_path: str,
) -> str:
    parts = [_GENERATED_HEADER, ""]
    parts.append("# " + "=" * 70)
    parts.append("# 1. Inductor output code (generated kernels + ``call`` entry point)")
    parts.append("# " + "=" * 70)
    # The whole computation (forward, or forward+loss+backward) is one graph, so
    # the single runnable Inductor ``call`` is inlined here.
    parts.append(inductor_chunks[0])
    parts.append("")
    parts.extend(_build_metadata_section(compiled, weights_path))
    parts.extend(_build_wrapper_section(compiled))
    parts.append("# " + "=" * 70)
    parts.append("# 4. Composition + runtime calling-convention wrapper")
    parts.append("# " + "=" * 70)
    parts.append(_DRIVER_SOURCE)
    return "\n".join(parts)


_DRIVER_SOURCE = (
    """\
_HELPER_NAMES = """
    + repr(_WRAPPER_HELPER_NAMES)
    + """
_CHAIN_NAMES = """
    + repr(_WRAPPER_CHAIN_NAMES)
    + '''

_blob = None
_params_buffers = None
_runtime_fn = None


def _load_blob():
    global _blob
    if _blob is None:
        _blob = _torch.load(WEIGHTS_PATH, weights_only=False)
    return _blob


def _load_params_buffers():
    return _load_blob()["param_buffer_flat"]


def _build_runtime():
    """Recompose the captured AOTAutograd wrappers around the Inductor ``call``.

    Each wrapper's globals are reconstructed from its plan: ``INNER`` binds the
    running composed callable, ``CALL`` the Inductor entry point, ``REC:<name>``
    a previously built wrapper (mutation/alias helpers), ``MOD:<name>`` a module,
    and everything else (subclass metadata, helper fns) is loaded as a residual
    global from the cache. The orchestration wrapper receives its inner callable
    as a parameter, with a no-op profiling hook and a null first-call context.
    """
    residuals = _load_blob().get("wrapper_residuals", {})
    by_name = {r["name"]: r for r in _RECORDS}
    built = {}

    def build_one(rec, inner):
        g = dict(residuals.get(rec["name"], {}))
        for key, tag in rec["plan"]:
            if tag == "INNER":
                g[key] = inner
            elif tag == "CALL":
                g[key] = call  # noqa: F821  (inlined Inductor entry point)
            elif tag.startswith("REC:"):
                g[key] = built[tag[4:]]
            elif tag.startswith("MOD:"):
                g[key] = _importlib.import_module(tag[4:])
        loc = {}
        exec(compile(rec["source"], "<" + rec["name"] + ">", "exec"), g, loc)
        return loc[rec["fn_name"]]

    for name in _HELPER_NAMES:
        if name in by_name:
            built[name] = build_one(by_name[name], None)

    inner = call  # noqa: F821
    for name in _CHAIN_NAMES:
        if name not in by_name:
            continue
        rec = by_name[name]
        if name == "runtime_wrapper_orchestration":
            fn = build_one(rec, None)

            def _driver(args, _fn=fn, _inner=inner):
                return _fn(_inner, _contextlib.nullcontext, lambda: None, args)

            inner = _driver
        else:
            inner = build_one(rec, inner)
        built[name] = inner
    return inner


def forward(*args, **kwargs):
    """Run the compiled model on user inputs.

    The lifted parameters/buffers are prepended to the flattened user inputs to
    form the boxed flat list ``[*params, *buffers, *user_inputs]`` that the
    composed runtime function expects; it is run under no_grad (this is an
    inference artifact) and its flat outputs are unflattened to the original
    output structure.
    """
    global _runtime_fn, _params_buffers
    if _runtime_fn is None:
        _runtime_fn = _build_runtime()
    if _params_buffers is None:
        _params_buffers = _load_params_buffers()
    user_flat, _ = _pytree.tree_flatten((args, kwargs))
    flat = [*_params_buffers, *user_flat]
    with _torch.no_grad():
        out = _runtime_fn(list(flat))
    return _pytree.tree_unflatten(list(out), _pytree.treespec_loads(OUT_SPEC))


if __name__ == "__main__":
    _pb = _load_params_buffers()
    print("Loaded", len(_pb), "parameter/buffer tensors from", WEIGHTS_PATH)
    print("Composed", len(_RECORDS), "AOTAutograd runtime wrapper(s).")
    print("forward() is ready; call it with the model's user inputs.")
'''
)


def _innermost_compiled_fn(artifact: Any) -> Any:
    """Walk ``__wrapped__`` to the innermost callable (the Inductor boxed call)."""
    fn = getattr(artifact, "_compiled_fn", None) or getattr(
        artifact, "inner_fn", artifact
    )
    seen: set[int] = set()
    while True:
        nxt = getattr(fn, "__wrapped__", None)
        if nxt is None or id(nxt) in seen:
            return fn
        seen.add(id(nxt))
        fn = nxt


def _classify_globals(
    globals_dict: dict[str, Any],
    call_id: int,
    id_to_name: dict[int, str],
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Split a wrapper's globals into a serializable plan and residual values.

    The plan records, per global name, how to rebind it at load time: ``INNER``
    (the running composed callable), ``CALL`` (the Inductor entry point),
    ``REC:<name>`` (another captured wrapper), or ``MOD:<name>`` (a module).
    Everything else (subclass metadata, helper functions, dim lists) is a
    residual value carried in the cache and reconstructed by unpickling.
    """
    import types as _types

    plan: list[tuple[str, str]] = []
    residual: dict[str, Any] = {}
    for key, value in globals_dict.items():
        if key == "__builtins__":
            continue
        if key in ("compiled_fn", "_compiled_fn_"):
            plan.append((key, "INNER"))
        elif id(value) == call_id:
            plan.append((key, "CALL"))
        elif id(value) in id_to_name:
            plan.append((key, "REC:" + id_to_name[id(value)]))
        elif isinstance(value, _types.ModuleType):
            plan.append((key, "MOD:" + value.__name__))
        else:
            residual[key] = value
    return plan, residual


def _capture_compiled_wrappers(
    gm: torch.fx.GraphModule, flat_args: list[Any]
) -> tuple[Any, list[dict[str, Any]]]:
    """Compile ``gm`` while capturing every codegen'd AOTAutograd wrapper.

    AOTAutograd notifies ``_codegen_exec.capture_generated_wrappers`` observers of
    each generated wrapper; we record ``(artifact_name, source, fn_name, globals,
    fn)`` and classify the globals against the final composition so the wrappers
    can be re-emitted and re-composed in the exported python. The graph is lowered
    like inference (any backward is already in the graph as ops).
    """
    from torch._functorch._aot_autograd import _codegen_exec
    from torch._inductor import standalone_compile

    raw: list[dict[str, Any]] = []

    def observer(artifact_name, source, fn_name, globals_dict, fn):
        raw.append(
            {
                "name": artifact_name,
                "fn_name": fn_name,
                "source": source,
                "globals": globals_dict,
                "fn": fn,
            }
        )

    with (
        _codegen_exec.capture_generated_wrappers(observer),
        torch.no_grad(),
    ):
        artifact = standalone_compile(
            gm, flat_args, dynamic_shapes="from_example_inputs"
        )

    call_id = id(_innermost_compiled_fn(artifact))
    id_to_name = {id(r["fn"]): r["name"] for r in raw}
    records = []
    for r in raw:
        plan, residual = _classify_globals(r["globals"], call_id, id_to_name)
        records.append(
            {
                "name": r["name"],
                "fn_name": r["fn_name"],
                "source": r["source"],
                "plan": plan,
                "residual": residual,
            }
        )
    return artifact, records


class NonStrictCompiled:
    """Internal holder for a precompiled computation / a loaded runnable."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        decompositions: dict | None = None,
    ) -> None:
        # ``fn`` is the whole computation: an nn.Module, or a callable that closes
        # over the module(s) it uses (e.g. ``lambda x: model(x)``, or a training
        # step that computes a loss and torch.autograd.grad).
        self._fn = fn
        self._decompositions = decompositions
        self._artifact: Any = None
        self._param_names: list[str] = []
        self._buffer_names: list[str] = []
        self._param_buffer_flat: list[Any] = []
        self._num_params_buffers: int = 0
        self._in_spec: pytree.TreeSpec | None = None
        self._out_spec: pytree.TreeSpec | None = None
        self._gm: torch.fx.GraphModule | None = None
        # Inductor output code chunk(s) and the captured AOTAutograd wrapper
        # records, populated by _compile() and re-emitted into the python artifact.
        self._inductor_chunks: list[str] = []
        self._wrapper_records: list[dict[str, Any]] = []
        # Set only on the load() path, where we wrap a reconstructed callable.
        self._loaded_forward: Callable[..., Any] | None = None

    def _compile(self, args: tuple[Any, ...]) -> None:
        capture = _capture(self._fn, args, self._decompositions)
        artifact, records = _capture_compiled_wrappers(capture.gm, capture.flat_args)

        self._artifact = artifact
        self._wrapper_records = records
        self._inductor_chunks = self._extract_inductor_code()
        self._param_names = capture.param_names
        self._buffer_names = capture.buffer_names
        self._param_buffer_flat = capture.param_buffer_flat
        self._num_params_buffers = capture.num_params_buffers
        self._in_spec = capture.in_spec
        self._out_spec = capture.out_spec
        self._gm = capture.gm

    def __call__(self, *args: Any) -> Any:
        # A NonStrictCompiled is runnable only after load(); precompile() itself
        # returns (python_code, cache) rather than a runnable.
        if self._loaded_forward is None:
            raise PrecompileError(
                "this object is not runnable; build one with "
                "torch.precompile.load(python_code, cache)."
            )
        return self._loaded_forward(*args)

    def _extract_inductor_code(self) -> list[str]:
        """Unpack the artifact and return the Inductor output module(s).

        Of the ``.py`` files Inductor emits we want the one defining the module-
        level ``call`` entry point (``call = runner.call``); the others are
        compile-time autotuning helpers not needed at runtime. The trailing
        ``__main__`` benchmark block is stripped so it is a clean module exposing
        ``call``.
        """
        if self._artifact is None:
            raise PrecompileError("nothing to extract; not compiled.")
        chunks: list[str] = []
        with tempfile.TemporaryDirectory() as unpack_dir:
            self._artifact.save(path=unpack_dir, format="unpacked")
            for root, _dirs, files in os.walk(unpack_dir):
                for name in sorted(files):
                    if not name.endswith(".py"):
                        continue
                    with open(os.path.join(root, name)) as f:
                        text = f.read()
                    if "def call(" in text and "call = runner.call" in text:
                        marker = '\nif __name__ == "__main__":'
                        idx = text.find(marker)
                        if idx != -1:
                            text = text[:idx].rstrip() + "\n"
                        chunks.append(text)
        if not chunks:
            raise PrecompileError(
                "could not locate the runnable Inductor output code for this artifact."
            )
        return chunks

    def parameters(self) -> list[Any]:
        """Return the lifted parameter tensors (e.g. to build an optimizer)."""
        return self._param_buffer_flat[: len(self._param_names)]

    def to_python_code(self) -> str:
        """Return the self-contained, executable Python artifact as a string.

        It embeds the Inductor output code, the captured AOTAutograd runtime
        wrappers, and a ``forward()`` that loads the parameter/buffer values from
        the companion cache (``WEIGHTS_PATH``; set via the env var for standalone
        exec). ``torch.precompile.load`` provides the cache directly.
        """
        assert self._artifact is not None
        return _build_python_source(self, self._inductor_chunks, _DEFAULT_WEIGHTS_PATH)

    def to_cache_bytes(self) -> bytes:
        """Return the binary cache as bytes.

        Holds parameter/buffer values, the residual wrapper globals (subclass
        metadata, dim lists, etc.), and the real Inductor/AOTAutograd
        compiled-artifact bytes. On load that artifact primes the inductor cache
        and is reconstructed via a FxGraphCache hit, so reload does not
        re-trace/re-lower the graph and (on GPU) restores bundled Triton kernels.
        """
        assert self._in_spec is not None and self._out_spec is not None
        wrapper_residuals = {
            rec["name"]: rec["residual"] for rec in self._wrapper_records
        }
        blob: dict[str, Any] = {
            "param_buffer_flat": self._param_buffer_flat,
            "param_names": self._param_names,
            "buffer_names": self._buffer_names,
            "num_params_buffers": self._num_params_buffers,
            "in_spec": pytree.treespec_dumps(self._in_spec),
            "out_spec": pytree.treespec_dumps(self._out_spec),
            "wrapper_residuals": wrapper_residuals,
            # None if the artifact is not serializable (uncacheable graph); load()
            # then falls back to executing the self-contained python.
            "artifact_bytes": self._try_artifact_binary_bytes(),
        }
        buf = io.BytesIO()
        torch.save(blob, buf)
        return buf.getvalue()

    def _try_artifact_binary_bytes(self) -> bytes | None:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            tmp = tf.name
        try:
            self._artifact.save(path=tmp, format="binary")
            with open(tmp, "rb") as f:
                return f.read()
        except Exception:
            return None
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def _make_cached_forward(
    artifact_bytes: bytes,
    params: list[Any],
    out_spec: pytree.TreeSpec,
) -> Callable[..., Any]:
    """Reconstruct the compiled artifact from the cache and drive it.

    ``CompiledArtifact.load`` primes the inductor cache from ``artifact_bytes``
    (so the graph is not re-traced/re-lowered and bundled kernels are restored)
    and rebuilds the full AOTAutograd runtime (subclass aware). The graph is
    functional, so it runs under no_grad.
    """
    from torch._inductor import CompiledArtifact

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tmp = tf.name
        tf.write(artifact_bytes)
    try:
        artifact = CompiledArtifact.load(path=tmp, format="binary")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    def forward(*args: Any, **kwargs: Any) -> Any:
        user_flat, _ = pytree.tree_flatten((args, kwargs))
        full_flat = [*params, *user_flat]
        with torch.no_grad():
            out_flat = artifact(*full_flat)
        return pytree.tree_unflatten(list(out_flat), out_spec)

    return forward


def _make_inlined_forward(
    python_code: str, cache: bytes, params: list[Any]
) -> Callable[..., Any]:
    """Fallback: execute the self-contained python string (recompiles kernels)."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        cache_path = tf.name
        tf.write(cache)
    module_ns: dict[str, Any] = {"__name__": "_precompiled_artifact"}
    exec(compile(python_code, "<precompile>", "exec"), module_ns)
    module_ns["WEIGHTS_PATH"] = cache_path
    # Share one parameter list so backward-populated grads land on the tensors
    # returned by parameters() (which an optimizer steps).
    module_ns["_params_buffers"] = params
    return module_ns["forward"]


def precompile(
    fn: Callable[..., Any],
    *args: Any,
    decompositions: dict | None = None,
) -> tuple[str, bytes]:
    """Ahead-of-time precompile ``fn`` against example ``args`` via make_fx.

    Returns ``(python_code, cache)`` -- a self-contained, executable Python source
    string and a binary cache (params + the real Inductor/AOTAutograd artifact).
    Reload a runnable with ``torch.precompile.load(python_code, cache)``.

    ``fn`` is the whole computation, e.g.::

        python_code, cache = torch.precompile(lambda model, x: model(x), model, x)


        def train_step(model, x, t):
            loss_fn(model(x), t).backward()  # or return autograd.grad(...)


        python_code, cache = torch.precompile(train_step, model, x, t)

    Among ``args``, the ``nn.Module`` arguments have their params/buffers lifted to
    graph inputs (nothing is baked); the rest are the runtime inputs, which is what
    the reloaded callable is invoked with (the modules are not passed again). If
    ``fn`` ran a backward, the resulting parameter gradients are harvested and
    appended as outputs; apply them to the reloaded ``parameters()`` via an
    optimizer.
    """
    torch._C._log_api_usage_once("torch.precompile")
    compiled = NonStrictCompiled(fn, decompositions=decompositions)
    compiled._compile(args)
    return compiled.to_python_code(), compiled.to_cache_bytes()


def _load(python_code: str, cache: bytes) -> NonStrictCompiled:
    """Reconstruct a runnable from ``(python_code, cache)`` produced by precompile.

    When the cache holds the serialized artifact (the common case) it is rebuilt
    via ``CompiledArtifact.load`` -- priming the inductor cache (FxGraphCache hit,
    no re-lowering; restores bundled kernels) and the full AOTAutograd runtime, so
    tensor subclasses (DTensor) work. Otherwise it falls back to executing the
    self-contained ``python_code`` (recompiling kernels from the inlined source).
    Call the result with the runtime inputs (the non-module args).
    """
    # Unpickling the cache references classes in AOTAutograd's runtime; import
    # dynamo first so that import completes in a non-circular order (otherwise a
    # cold load can hit a runtime_wrappers <-> _dynamo circular import).
    import torch._dynamo

    blob = torch.load(io.BytesIO(cache), weights_only=False)
    params = blob["param_buffer_flat"]
    out_spec = pytree.treespec_loads(blob["out_spec"])

    artifact_bytes = blob.get("artifact_bytes")
    if artifact_bytes is not None:
        forward = _make_cached_forward(artifact_bytes, params, out_spec)
    else:
        forward = _make_inlined_forward(python_code, cache, params)

    obj = NonStrictCompiled.__new__(NonStrictCompiled)
    obj._fn = None  # type: ignore[assignment]
    obj._decompositions = None
    obj._artifact = None
    obj._inductor_chunks = []
    obj._wrapper_records = []
    obj._param_names = blob["param_names"]
    obj._buffer_names = blob["buffer_names"]
    obj._param_buffer_flat = params
    obj._num_params_buffers = blob["num_params_buffers"]
    obj._in_spec = pytree.treespec_loads(blob["in_spec"])
    obj._out_spec = out_spec
    obj._gm = None
    obj._loaded_forward = forward
    return obj


# Allow ``torch.precompile.load(python_code, cache)``.
precompile.load = _load  # type: ignore[attr-defined]
