# Single-bubble reference data

`legacy_single_bubble_reference.npz` contains reference trajectories
generated from the uncoupled Keller–Miksis implementation in
`MultiBubble1D(3).ipynb`.

The fixture uses the material-property and nondimensionalization
conventions of the legacy notebook and contains three driven
single-bubble cases ranging from weak to strongly nonlinear response.

The data are used only for regression verification of the NumPy
reference implementation. They are independent of the current
`single_bubble_rhs` implementation.