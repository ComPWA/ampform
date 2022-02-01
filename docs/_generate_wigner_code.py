# pylint:disable=import-outside-toplevel, too-many-locals
def generate_wigner_code() -> None:
    import inspect
    import logging
    import textwrap

    import black
    import qrules
    import sympy as sp

    from ampform.kinematics import (
        compute_wigner_angles,
        create_four_momentum_symbols,
    )

    logging.getLogger().setLevel(logging.ERROR)

    topologies = qrules.topology.create_isobar_topologies(3)
    topology = topologies[0]
    assert topology.get_edge_ids_outgoing_from_node(1) == {1, 2}
    state_id = 1

    momenta = create_four_momentum_symbols(topology)
    angle_definitions = compute_wigner_angles(topology, momenta, state_id)
    _, beta, _ = map(sp.Symbol, angle_definitions)
    beta_expr = angle_definitions[str(beta)]

    func = sp.lambdify(momenta.values(), beta_expr.doit(), cse=True)

    src = inspect.getsource(func)
    import_statements = """
    from numpy import arccos, array, einsum, greater, nan, ones, select, sqrt, sum, zeros
    """.strip()
    src = import_statements + "\n" + src
    src = black.format_str(src, mode=black.Mode(line_length=145))

    filename = "generated-code"
    with open(f"usage/helicity/{filename}.py", "w") as stream:
        stream.write(src)

    md_src = f"""\
    # Generated code

    This notebook renders the {{mod}}`numpy` code for Wigner rotation angles as
    described in
    {{ref}}`usage/helicity/spin-alignment:Compute Wigner rotation angles`.

    ```{{literalinclude}} {filename}.py
    ---
    class: full-width
    language: python
    ---
    ```
    """
    md_src = textwrap.dedent(md_src)
    with open(f"usage/helicity/{filename}.md", "w") as stream:
        stream.write(md_src)
