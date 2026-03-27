# Tasks for Feature F12

## T01 — verify backward_pass correctness
**Status**: not done
**Description**: Confirm backward_pass(g) produces correct gradients for a turkey_feather graph with w1=1000, w2=3000. Expected: grad_w1=1875, grad_w2=3500. Add test cases if missing.

## T02 — verify mermaid_viz gradient display
**Status**: not done
**Description**: Confirm mermaid_html(g, show_gradients=True, highlighted) renders gradient values on nodes. Add test cases to test_mermaid_viz.py if missing.

## T03 — add backward pass lesson cell in main.py
**Status**: not done
**Description**: Add a new lesson section. Show the completed forward-pass table (all turkeys, step=4). Add a "Run Backward Pass" button. When clicked, run backward_pass on turkey 1's graph and display mermaid_html with show_gradients=True.

## T04 — add prose cell connecting gradients to weight update
**Status**: not done
**Description**: Add a markdown cell explaining: grad_w1=1875 and grad_w2=3500 are both positive, so decrease both weights. Show the update rule: new_w = w - learning_rate × gradient. Show one step with learning_rate=0.01.

## T05 — update main_wasm.py and redeploy
**Status**: not done
**Description**: Inline the updated lesson code into main_wasm.py and re-export. Run: uv run marimo export html-wasm src/main_wasm.py -o docs --mode run -f

## T06 — write tests
**Status**: not done
**Description**: Test backward_pass gives correct gradients for starting weights. Test mermaid_html with show_gradients=True contains gradient values in the output string.
