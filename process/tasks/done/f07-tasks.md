# Tasks for Feature F07

## T01 — create mermaid_viz.py
**Status**: not done
**Description**: Write src/mermaid_viz.py with two functions: build_mermaid(g, show_gradients, highlighted) returns a Mermaid diagram string; mermaid_html(g, show_gradients, highlighted) wraps it in a full HTML string with the Mermaid CDN script tag, ready for mo.Html().

## T02 — wire mermaid_viz into main.py
**Status**: not done
**Description**: Replace draw_graph calls in main.py with mo.Html(mermaid_html(...)).

## T03 — write tests
**Status**: not done
**Description**: Test that build_mermaid returns a string containing expected node ids and edge arrows for a turkey_feather graph. Test that mermaid_html wraps it in an html tag.
