import dolfinx as dfx
import ufl


def mass_form(
    V: dfx.fem.FunctionSpace,
    u: dfx.fem.function.Function | ufl.argument.Argument,
) -> ufl.form.Form:
    v = ufl.TestFunction(V)
    return u * v * ufl.dx(domain=V.mesh)


def stiffness_form(
    V: dfx.fem.FunctionSpace,
    u: dfx.fem.function.Function | ufl.argument.Argument,
    kappa_function: dfx.fem.function.Function,
) -> ufl.form.Form:
    v = ufl.TestFunction(V)
    return (
        kappa_function
        * ufl.dot(ufl.grad(u), ufl.grad(v))
        * ufl.dx(domain=V.mesh)
    )
