# Spec for back

* An interactive Marimo notebook for readers who have just finished Chapter 4 of "How They Think" (Eric Silberstein)
* Guides the reader through backpropagation via stepped exploration, not passive reading
* Uses the book's turkey feather model as the entry example, then offers progressively larger networks
* Each section reveals one new concept — controls appear only when the narrative is ready for them
* The reader manipulates weights via sliders and observes how forward pass values and gradients respond in real time
* Computation graphs are drawn with matplotlib: nodes labeled with current values, edges labeled with gradients
* Three example networks are provided: small (2 weights, turkey feather), medium (4-6 weights), large (one full hidden layer)
* The notebook is reactive: changing a slider immediately recomputes and redraws everything downstream
* No prior coding knowledge required to use; the reader only interacts with sliders and buttons
