# Higra — CLAUDE.md

Higra v0.6.13. C++17/Python library for hierarchical graph analysis (trees, watersheds, ultrametrics).
Maintainer: Benjamin Perret (PerretB, ESIEE Paris). License: CECILL-B.

---

## Design Philosophy

Higra is built around two core ideas:

**1. The hierarchy as first-class citizen.**
Every algorithm in Higra either builds a hierarchy (tree) or operates on one. The canonical representation is a Binary Partition Tree (BPT) paired with an altitude array. The tree is just a parent index array; altitudes are separate. This clean separation allows any accumulator or attribute function to work generically across all hierarchy types.

**2. Zero-cost C++ abstractions exposed through a thin Python layer.**
All computationally heavy code lives in header-only C++ templates (`include/higra/`). The Python layer (`higra/`) is thin: it validates inputs, handles the image-graph linearization/delinearization, and delegates to C++ via pybind11. There is no "Python fallback" for performance-critical paths.

These two principles shape every design decision in the codebase: if a proposed implementation breaks either of them, reconsider the approach.

---

## Repository Structure

```
include/higra/          # C++ header-only algorithms (templates, namespace hg)
  algo/                 #   graph/tree algorithms (watershed, tree cuts, rag, ...)
  hierarchy/            #   hierarchy construction (bpt_canonical, quasi-flat zones, ...)
  accumulator/          #   tree/graph accumulators
  attribute/            #   tree node attributes
  structure/            #   graph and tree data structures, union-find, LCA
  config.hpp            #   version defines
  utils.hpp             #   index_t, hg_assert macros, HG_TRACE, HG_TEMPLATE_* macros

higra/                  # Python package (mirrors include/higra/ structure)
  __init__.py           #   public API: re-exports everything with star imports
  pymodule.cpp          #   PYBIND11_MODULE entry point, registers all submodules
  all.hpp               #   includes all py_*.hpp binding headers
  py_common.hpp         #   add_type_overloads<> template for type dispatch
  algo/                 #   each module: py_<mod>.hpp + py_<mod>.cpp + <mod>.py + __init__.py
  hierarchy/
  accumulator/
  assessment/
  attribute/
  image/
  interop/
  io_utils/
  structure/

test/
  python/               #   pytest tests, mirrors higra/ (test_algo/, test_hierarchy/, ...)
  cpp/                  #   C++ catch2 unit tests

lib/                    #   vendored C++ deps: xtensor, pybind11, TBB, xtl, xsimd, xtensor-python
doc/                    #   Sphinx documentation
poubelle/               #   scratch scripts and MREs, never committed to main
```

---

## Binding Pattern

Every module follows an identical 4-file pattern. Example for `algo/watershed`:

```
include/higra/algo/watershed.hpp    # C++ algorithm, templates, namespace hg
higra/algo/py_watershed.hpp         # declares void py_init_watershed(pybind11::module &m)
higra/algo/py_watershed.cpp         # pybind11 bindings, calls add_type_overloads<>
higra/algo/watershed.py             # Python wrapper: input validation, linearize/delinearize, docstring
higra/algo/__init__.py              # re-exports the Python wrapper functions
```

`higra/pymodule.cpp` calls every `py_init_*` function. `higra/all.hpp` includes every `py_*.hpp`. When adding a new function, all four files must be updated, plus the module's `__init__.py`.

The compiled extension module is named `higram` (with a trailing `m`). It is aliased as `higra.cpp` inside `__init__.py` (`from . import higram as cpp`). Binding functions are prefixed with `_` (e.g. `hg.cpp._labelisation_watershed`); the unprefixed public function lives in the Python wrapper.

---

## C++ Conventions (PerretB style)

**File header.** Every new file starts with the CECILL-B copyright block:
```cpp
/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/
```

**Function signature pattern.** Every public C++ function follows this structure:
```cpp
template<typename graph_t, typename T>
auto my_function(const graph_t &graph, const xt::xexpression<T> &xedge_weights) {
    HG_TRACE();                              // always first line
    auto &edge_weights = xedge_weights.derived_cast();  // materialize lazy expression
    hg_assert_edge_weights(graph, edge_weights);        // validate shape
    hg_assert_1d_array(edge_weights);                   // validate dimensionality
    // ...
}
```

**Key macros** (defined in `include/higra/utils.hpp`):

| Macro | Purpose |
|---|---|
| `HG_TRACE()` | Profiling/logging hook, every public function entry |
| `hg_assert(cond, msg)` | Runtime assertion (no-op in release builds) |
| `hg_assert_edge_weights(g, w)` | Checks shape[0] == num_edges(g) |
| `hg_assert_node_weights(t, w)` | Checks shape[0] == num_vertices(t) |
| `hg_assert_1d_array(a)` | Checks dimension() == 1 |
| `hg_assert_integral_value_type(a)` | Static assert on value type |
| `HG_TEMPLATE_NUMERIC_TYPES` | Expands to all numeric types for `add_type_overloads<>` |
| `HG_TEMPLATE_FLOAT_TYPES` | `float, double` |
| `HG_TEMPLATE_INTEGRAL_TYPES` | All int/uint variants |

**Array types.** Use `array_1d<T>`, `array_2d<T>` etc. (Higra's xtensor aliases). For output allocation: `array_1d<T>::from_shape({n})` or `xt::empty<T>({n})`. Force evaluation of lazy expressions with `xt::eval(...)` before returning.

**Index type.** Always use `hg::index_t` (alias for `int64_t`) for vertex/edge indices. The sentinel value for "no index" is `hg::invalid_index` (-1).

**Type dispatch in bindings.** Use `add_type_overloads<MyStruct, HG_TEMPLATE_NUMERIC_TYPES>(m, doc)` to register one binding per numeric type. This is how Higra avoids runtime type checks in C++.

**Style preferences observed in PerretB's code:**
- Prefer free functions over methods; stateful classes are the exception, not the rule.
- Lambdas are used liberally for local logic (e.g. the `stream` lambda in `labelisation_watershed`).
- No raw `new`/`delete`; use stack allocation and xtensor arrays.
- Comments in C++ are sparse and algorithmic (reference to the paper, not "this loop iterates").
- Return types use `auto` (deduced) rather than explicit types when non-trivial.

---

## Python Conventions

**Docstring format.** NumPy/Sphinx style with a mandatory `:Complexity:` section for non-trivial algorithms:
```python
def my_function(graph, edge_weights):
    """
    Short description.

    Reference to paper if applicable.

    :Complexity:

    :math:`\mathcal{O}(n \log n)` with :math:`n` the number of edges.

    :param graph: input graph
    :param edge_weights: Weights on the edges of the graph
    :return: description
    """
```

**Image-graph compatibility.** Graph vertices can represent pixels in a 2D/3D image. Vertex weight arrays may be shaped `(H, W)` or `(H, W, C)` instead of `(N,)`. Always call `linearize_vertex_weights` before passing to C++, and `delinearize_vertex_weights` on the output. Edge weights are always 1D and do not require this treatment.

**Input validation.** Do Python-level type checks before calling into C++. Example pattern:
```python
if not issubclass(vertex_seeds.dtype.type, np.integer):
    raise ValueError("vertex_seeds must be an array of integers")
vertex_seeds = hg.cast_to_dtype(vertex_seeds, np.int64)
```

**Data cache.** Higra has a global weak-reference cache (`hg.get_attribute`, `hg.set_attribute`) that associates computed attributes with graph/tree objects. Use it when an attribute is expensive to recompute and likely to be reused (e.g. `attribute_volume`). Consult existing attribute functions for the pattern.

---

## Algorithm Design Principles

These are the questions PerretB asks during code review. Answer them before submitting any algorithm change:

**1. Is the operation correctly batched?**
If a function is called inside a loop over elements (seeds, vertices, edges), ask whether the loop can be inverted: process the full batch at once, then update. Per-element relabeling is O(k * |component|); batch relabeling is O(|component|). This is not just a performance concern — sequential per-element processing can produce incorrect intermediate states when elements share components.

**2. Are there asymmetric cases?**
Some operations are not symmetric. Seed addition and seed removal in an incremental watershed have fundamentally different complexity profiles: addition only grows cuts, removal may merge components and require a reset + re-seed. Each direction needs its own correctness analysis.

**3. Can a cached auxiliary structure reduce the work?**
Before implementing a brute-force solution, check whether the BPT, the MST, the union-find, or the LCA structure already encodes the information needed. The BPT `visitCount` array, for example, directly identifies which subtree a seed belongs to and can tighten BFS scope during removal.

**4. What happens with simultaneous multi-element operations?**
When processing a batch (e.g. removing k seeds at once), consider: do any of the k elements share a component after the operation? If yes, a per-element loop will visit overlapping regions multiple times and may produce different results depending on processing order. The correct approach is to identify all affected components first, then update them jointly.

**5. Does the algorithm reference a published paper?**
Higra algorithms are always tied to a peer-reviewed reference. If implementing or modifying an algorithm, cite the paper in the C++ doxygen block and in the Python docstring.

---

## Pre-Commit Discipline

**Before `git add`:**
- Run `git diff` and read every changed line.
- Check for CRLF line endings (Windows editors produce `\r\n`; Higra uses LF). Visible as `^M` in the diff. Fix with `git config core.autocrlf input` or a `.gitattributes` rule.
- Verify that no debug print statements, commented-out code, or TODO comments are included.
- Confirm that every new C++ file has the CECILL-B header and `HG_TRACE()` in every public function.

**Before opening a PR:**
- Run the full Python test suite: `python -m pytest test/python/ -x -q`
- Check that the new test covers both the happy path and relevant edge cases (empty input, single element, batch of identical elements).
- Re-read the diff with the algorithm design questions above in mind. The maintainer will ask them.

---

## Open Issues (as of 2026)

| # | Title | Nature |
|---|---|---|
| #285 | Upgrade TBB | C++ dependency; current TBB uses removed APIs, blocks ARM Mac |
| #245 | C++17 + nanobind bindings | Major refactor; blocked on xtensor-python migration |
| #236 | ETE Toolkit interop | Python only; PerretB has a working POC in the issue thread |
| #182 | Improve graph I/O | Pink format is text-based and slow; suggestion: HDF5 via h5py |
| #12 | sklearn interop | Pure Python; expose clustering via `BaseEstimator` + `ClusterMixin` |

---

## Key Entry Points for Common Tasks

| Task | Start here |
|---|---|
| Modify a watershed algorithm | `include/higra/algo/watershed.hpp` + `higra/algo/py_watershed.cpp` |
| Modify a hierarchy algorithm | `include/higra/hierarchy/` + `higra/hierarchy/py_*.cpp` |
| Add a tree attribute | `include/higra/attribute/` + `higra/attribute/` |
| Add an external library interop | `higra/interop/` |
| Fix graph I/O | `higra/io_utils/` |
| Register a new C++ binding | `higra/pymodule.cpp` + `higra/all.hpp` |
| Add a new public Python function | module `__init__.py` re-export + `higra/__init__.py` star import chain |
