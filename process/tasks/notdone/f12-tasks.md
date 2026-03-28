# Tasks for Feature F12

## T01 — test backward_pass with turkey dataset
**Status**: not done
**Description**: Add tests to test_compgraph.py: backward_pass on Turkey 1 (w1=1000, w2=3000) gives grad_w1=1000, grad_w2=1500. Sum of grad_w1 across all 3 turkeys = 1875; sum of grad_w2 = 3500.

## T02 — test mermaid_html with show_gradients=True
**Status**: not done
**Description**: Add test to test_mermaid_viz.py: build_mermaid with show_gradients=True contains "grad:" in the output string.

## T03 — add backward pass lesson cell in main.py
**Status**: not done
**Description**: Add prose intro cell and interactive cell. Show completed forward-pass table (step=4). Add "Run Backward Pass" button. When clicked, run forward+backward pass on all 3 turkey graphs, display Turkey 1 Mermaid with show_gradients=True.

## T04 — add prose cell connecting gradients to weight update
**Status**: not done
**Description**: Add markdown cell: summed grad_w1=1875 and grad_w2=3500 match F11. Show weight update rule: new_w = w - lr × gradient. Show one step with lr=0.01: w1=981.25, w2=2965.

## T05 — update main_wasm.py and redeploy
**Status**: not done
**Description**: Inline updated lesson code into main_wasm.py and re-export. Run: uv run marimo export html-wasm src/main_wasm.py -o docs --mode run -f
