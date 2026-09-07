import math
import ufl 
from typing import Tuple
import dolfinx as dfx


def ic_pulse2D(mu:float,s:float,b:float,factor:float,kappa_value:float,x0:ufl.SpatialCoordinate,x1:ufl.SpatialCoordinate)-> Tuple[ufl.Form, ufl.Form]:
    _phase = math.ceil((mu + s) / b) % 2
    _k = math.ceil((mu + s) / b) // 2
    base_function_1D = ufl.conditional(
        abs(x0 - mu) < s,
        factor * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 3,
        0,
    ) 
    base_function_derivative_1D = ufl.conditional(
        abs(x0 - mu) < s,
        factor
        * 3
        * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 2
        * ufl.cos(ufl.pi * (x0 - (mu - s)) / (2 * s))
        * (ufl.pi / (2 * s)),
        0,
    )
    if _phase:
        u0_expr_1D = ufl.replace(base_function_1D, {x0: 2 * _k * b + x0}) - ufl.replace(base_function_1D, {x0: 2 * _k * b - x0})
        v0_expr_1D = -kappa_value * (
            ufl.replace(base_function_derivative_1D, {x0: 2 * _k * b + x0})
            - ufl.replace(base_function_derivative_1D, {x0: 2 * _k * b - x0})
        )
    else:
        u0_expr_1D = -ufl.replace(base_function_1D, {x0: 2 * _k * b - x0}) + ufl.replace(base_function_1D, {x0: 2 * (_k - 1) * b + x0})
        v0_expr_1D = -kappa_value * (-ufl.replace(base_function_derivative_1D, {x0: 2 * _k * b - x0})
            + ufl.replace(base_function_derivative_1D,{x0: 2 * (_k - 1) * b + x0},)
        )
    u0_expr = u0_expr_1D * ufl.replace(base_function_1D, {x0: x1}) + ufl.replace(u0_expr_1D,{x0:x1})*base_function_1D
    v0_expr = (
        v0_expr_1D * ufl.replace(base_function_1D, {x0: x1})
        + +ufl.replace(v0_expr_1D, {x0: x1}) * base_function_1D
    )
    return u0_expr, v0_expr


def interpolate_ufl_expression(
    V: dfx.fem.FunctionSpace, ufl_expr: ufl.Form
) -> dfx.fem.Function:
    func = dfx.fem.Function(V)
    df_expr = dfx.fem.Expression(ufl_expr, V.element.interpolation_points)
    func.interpolate(df_expr)
    return func


###
#  Below: Code for travelling pulse example. Not used in the current version
###



class TimeDependentUFLFunctor:
    """A Functor to represent time dependent functions"""

    def __init__(func, expression_lambda, domain, reevaluation=False):
        func.domain = domain
        func._expression_lambda = expression_lambda
        func.t = dfx.fem.Constant(domain, 0.0)
        func.expression = expression_lambda(func.t)
        func.reevaluation = reevaluation

    def copy(func):
        return TimeDependentUFLFunctor(
            func._expression_lambda, func.domain, reevaluation=func.reevaluation
        )

    def locked_evaluation(func, t_eval=0.0):
        return func._expression_lambda(t_eval)

    def __str__(func):
        expression_str = str(func.expression).replace(str(func.t), "t")
        return f"{expression_str} evaluated at t={func.t.value:04f}"

    def __call__(func, t_eval):
        func.t.value = t_eval
        if func.reevaluation:
            func.expression = func._expression_lambda(func.t)
        return func.expression


def rhs_tpulse_genFunctor(
    domain,
    x0,
    x1,
    kappa_value,
    factor,
    mu,
    s,
    b,
):
    base_function = ufl.conditional(
        abs(x0 - mu) < s,
        factor * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 3,
        0,
    )
    base_function_derivative = ufl.conditional(
        abs(x0 - mu) < s,
        factor
        * 3
        * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 2
        * ufl.cos(ufl.pi * (x0 - (mu - s)) / (2 * s))
        * (ufl.pi / (2 * s)),
        0,
    )
    base_function_derivative2 = ufl.conditional(
        abs(x0 - mu) < s,
        factor
        * 6
        * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s))
        * ufl.cos(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 2
        * (ufl.pi / (2 * s)) ** 2
        - factor
        * 3
        * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 2
        * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s))
        * (ufl.pi / (2 * s)) ** 2,
        0,
    )

    sol1D, deriv1D = traveling_pulse_ufl(
                x0, kappa_value, base_function, base_function_derivative, factor, mu, s, b
            )
    f = TimeDependentUFLFunctor(
        lambda t: -kappa_value**2
        * (
            sol1D(t) * ufl.replace(base_function_derivative2, {x0: x1})
            + ufl.replace(sol1D(t), {x0: x1}) * base_function_derivative2
        ),
        domain,
        reevaluation=True,  # this is expensive
    )
    return f

def u_exakt_tpulse_genFunctor(
    domain,
    x0,
    x1,
    kappa_value,
    factor,
    mu,
    s,
    b
):
    base_function = ufl.conditional(
        abs(x0 - mu) < s,
        factor * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 3,
        0,
    )
    base_function_derivative = ufl.conditional(
        abs(x0 - mu) < s,
        factor
        * 3
        * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 2
        * ufl.cos(ufl.pi * (x0 - (mu - s)) / (2 * s))
        * (ufl.pi / (2 * s)),
        0,
    )
    sol1D, deriv1D = traveling_pulse_ufl(
                x0, kappa_value, base_function, base_function_derivative, factor, mu, s, b
            )
    u_exakt = TimeDependentUFLFunctor(
        lambda t: sol1D(t) * ufl.replace(base_function, {x0: x1})
        + ufl.replace(sol1D(t), {x0: x1}) * base_function,
        domain,
        reevaluation=True,
    )
    return u_exakt


def v_exakt_tpulse_genFunctor(
    domain,
    x0,
    x1,
    kappa_value,
    factor,
    mu,
    s,
    b,
):
    base_function = ufl.conditional(
                abs(x0 - mu) < s,
                factor * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 3,
                0,
            )
    base_function_derivative = ufl.conditional(
        abs(x0 - mu) < s,
        factor
        * 3
        * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 2
        * ufl.cos(ufl.pi * (x0 - (mu - s)) / (2 * s))
        * (ufl.pi / (2 * s)),
        0,
    )
    sol1D, deriv1D = traveling_pulse_ufl(
        x0, kappa_value, base_function, base_function_derivative, factor, mu, s, b
    )
    v_exakt = TimeDependentUFLFunctor(
        lambda t: deriv1D(t) * ufl.replace(base_function, {x0: x1})
        + ufl.replace(deriv1D(t), {x0: x1}) * base_function,
        domain,
        reevaluation=True,
    )
    return v_exakt


def traveling_pulse_ufl(
      x, kappa_value, base_function, base_function_derivative, factor, mu, s, b
    ):
        import math

        right_support_bound = mu + s

        def sol(_t):
            phase = math.ceil((kappa_value * _t + right_support_bound) / b) % 2
            k = math.ceil((kappa_value * _t + right_support_bound) / b) // 2
            if phase:
                return ufl.replace(
                    base_function, {x: 2 * k * b + x - kappa_value * _t}
                ) - ufl.replace(base_function, {x: 2 * k * b - x - kappa_value * _t})
            else:
                return -ufl.replace(
                    base_function, {x: 2 * k * b - x - kappa_value * _t}
                ) + ufl.replace(
                    base_function, {x: 2 * (k - 1) * b + x - kappa_value * _t}
                )

        def derivative(_t):
            phase = math.ceil((kappa_value * _t + right_support_bound) / b) % 2
            k = math.ceil((kappa_value * _t + right_support_bound) / b) // 2
            if phase:
                return -kappa_value * (
                    ufl.replace(
                        base_function_derivative, {x: 2 * k * b + x - kappa_value * _t}
                    )
                    - ufl.replace(
                        base_function_derivative, {x: 2 * k * b - x - kappa_value * _t}
                    )
                )
            else:
                return -kappa_value * (
                    -ufl.replace(
                        base_function_derivative, {x: 2 * k * b - x - kappa_value * _t}
                    )
                    + ufl.replace(
                        base_function_derivative,
                        {x: 2 * (k - 1) * b + x - kappa_value * _t},
                    )
                )

        return sol, derivative
