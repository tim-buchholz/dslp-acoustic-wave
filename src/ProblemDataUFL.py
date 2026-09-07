from mpi4py import MPI
import numpy as np
import dolfinx as dfx
import ufl
from petsc4py import PETSc


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


class ProblemData:
    @classmethod
    def valid_numbers(cls):
        return [4, 9, 101, 104, 123, 125, 201, 204, 223, 228, 230]

    def __init__(
        self, example_number, V, wave_propagation_speed=1.0, provide_grad=False
    ) -> None:
        self.example_number = example_number
        self.dim = 1 if example_number < 200 else 2
        self.u_exakt, self.v_exakt, self.u0, self.v0, self.f, self.u_bc = [None] * 6
        self.c_init = wave_propagation_speed
        self.c = dfx.fem.Constant(V.mesh, wave_propagation_speed)
        self.solution_expression = None
        self.provide_grad = provide_grad
        self.grad_u0 = None
        self.grad_v0 = None

        x = ufl.SpatialCoordinate(V.mesh)

        if self.example_number == 0:
            self.u_exakt = None
            self.u_bc = None
            self.u0 = 0.0 * x[0]
            self.v_exakt = None
            self.v0 = 0.0 * x[0]
            self.f = TimeDependentUFLFunctor(
                lambda t: ((0.0 * x[0]) * t),
                V.mesh,
            )
        elif self.example_number == 4:  # a simple sin exp example
            self.u_exakt = TimeDependentUFLFunctor(
                lambda t: ufl.sin(3 * ufl.pi * x[0]) * ufl.exp(t), V.mesh
            )
            self.u0 = self.u_exakt.locked_evaluation()
            self.v_exakt = TimeDependentUFLFunctor(
                lambda t: ufl.sin(3 * ufl.pi * x[0]) * ufl.exp(t), V.mesh
            )
            self.v0 = self.v_exakt.locked_evaluation()
            self.f = TimeDependentUFLFunctor(
                lambda t: (1 + self.c * self.c * ufl.pi * ufl.pi * 9)
                * ufl.sin(3 * ufl.pi * x[0])
                * ufl.exp(t),
                V.mesh,
            )

        elif (
            self.example_number == 9
        ):  # example with inhom bc, good for testing time convergence
            self.u_exakt = TimeDependentUFLFunctor(
                lambda t: ufl.sin(0.5 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * t), V.mesh
            )
            self.u_bc = self.u_exakt.copy()
            self.u0 = self.u_exakt.locked_evaluation()
            self.v_exakt = TimeDependentUFLFunctor(
                lambda t: ufl.sin(0.5 * ufl.pi * x[0])
                * (-2 * ufl.pi * ufl.sin(2 * ufl.pi * t)),
                V.mesh,
            )
            self.v0 = self.v_exakt.locked_evaluation()
            self.f = TimeDependentUFLFunctor(
                lambda t: (
                    (-4 + self.c * self.c / 4)
                    * ufl.pi**2
                    * ufl.sin(0.5 * ufl.pi * x[0])
                    * ufl.cos(2 * ufl.pi * t)
                ),
                V.mesh,
            )
        elif self.example_number == 99:  # everything zero but right b.c.
            self.u_exakt = None
            self.u_bc = TimeDependentUFLFunctor(
                lambda t: ufl.conditional(abs(x[0]) > 0.5, 1.0 * x[0], 0.0 * x[0]),
                V.mesh,
            )
            self.u0 = 0.0 * x[0]
            self.v_exakt = None
            self.v0 = 0.0 * x[0]
            self.f = TimeDependentUFLFunctor(
                lambda t: ((0.0 * x[0]) * t),
                V.mesh,
            )
        elif self.example_number == 101:
            base = ufl.sin(ufl.pi * x[0]) ** 2
            base_xx = 2 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x[0])
            time_factor = lambda t: ufl.cos(2 * ufl.pi * t) ** 2 + 1

            self.u_exakt = TimeDependentUFLFunctor(
                lambda t: base * time_factor(t), V.mesh
            )
            self.u0 = self.u_exakt.locked_evaluation()
            self.v_exakt = TimeDependentUFLFunctor(
                lambda t: -2 * ufl.pi * base * ufl.sin(4 * ufl.pi * t), V.mesh
            )
            self.v0 = self.v_exakt.locked_evaluation()
            self.f = TimeDependentUFLFunctor(
                lambda t: -8 * ufl.pi**2 * base * ufl.cos(4 * ufl.pi * t)
                - self.c**2 * base_xx * time_factor(t),
                V.mesh,
            )
        elif self.example_number == 104:
            base = ufl.sin(ufl.pi * x[0]) ** 2
            base_xx = 2 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x[0])

            self.u_exakt = TimeDependentUFLFunctor(
                lambda t: base * ufl.exp(t), V.mesh
            )
            self.u0 = self.u_exakt.locked_evaluation()
            self.v_exakt = TimeDependentUFLFunctor(
                lambda t: base * ufl.exp(t), V.mesh
            )
            self.v0 = self.v_exakt.locked_evaluation()
            self.f = TimeDependentUFLFunctor(
                lambda t: (base - self.c**2 * base_xx) * ufl.exp(t),
                V.mesh,
            )
        elif self.example_number == 123:
            mu, s, b, x0 = 0.5, 0.2, 1.0, x[0]
            base_function = ufl.conditional(
                abs(x0 - mu) < s, ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 3, 0
            )

            base_function_derivative = ufl.conditional(
                abs(x0 - mu) < s,
                3
                * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 2
                * ufl.cos(ufl.pi * (x0 - (mu - s)) / (2 * s))
                * (ufl.pi / (2 * s)),
                0,
            )
            sol, derivative = self.traveling_pulse_ufl(
                x[0], base_function, base_function_derivative, mu, s, b
            )
            self.u_exakt = TimeDependentUFLFunctor(
                lambda t: sol(t), V.mesh, reevaluation=True
            )
            self.u0 = self.u_exakt.locked_evaluation()
            self.v_exakt = TimeDependentUFLFunctor(
                lambda t: derivative(t), V.mesh, reevaluation=True
            )
            self.v0 = self.v_exakt.locked_evaluation()
            self.f = TimeDependentUFLFunctor(
                lambda t: dfx.fem.Constant(V.mesh, 0.0), V.mesh
            )
        elif self.example_number == 125:
            mu1, mu2, s, b, x0 = 0.55, 0.45, 0.2, 1.0, x[0]
            base_function1 = ufl.conditional(
                abs(x0 - mu1) < s, ufl.sin(ufl.pi * (x0 - (mu1 - s)) / (2 * s)) ** 3, 0
            )
            base_function2 = ufl.conditional(
                abs(x0 - mu2) < s, ufl.sin(ufl.pi * (x0 - (mu2 - s)) / (2 * s)) ** 3, 0
            )
            base_function_derivative1 = ufl.conditional(
                abs(x0 - mu1) < s,
                3
                * ufl.sin(ufl.pi * (x0 - (mu1 - s)) / (2 * s)) ** 2
                * ufl.cos(ufl.pi * (x0 - (mu1 - s)) / (2 * s))
                * (ufl.pi / (2 * s)),
                0,
            )
            base_function_derivative2 = ufl.conditional(
                abs(x0 - mu2) < s,
                3
                * ufl.sin(ufl.pi * (x0 - (mu2 - s)) / (2 * s)) ** 2
                * ufl.cos(ufl.pi * (x0 - (mu2 - s)) / (2 * s))
                * (ufl.pi / (2 * s)),
                0,
            )

            sol1, derivative1 = self.traveling_pulse_ufl(
                x[0], base_function1, base_function_derivative1, mu1, s, b
            )
            sol2, derivative2 = self.traveling_pulse_ufl(
                x[0], base_function2, base_function_derivative2, mu2, s, b
            )
            self.u_exakt = TimeDependentUFLFunctor(
                lambda t: sol1(t) - sol2(t), V.mesh, reevaluation=True
            )
            self.u0 = self.u_exakt.locked_evaluation()
            self.v_exakt = TimeDependentUFLFunctor(
                lambda t: derivative1(t) - derivative2(t), V.mesh, reevaluation=True
            )
            self.v0 = self.v_exakt.locked_evaluation()
            self.f = TimeDependentUFLFunctor(
                lambda t: dfx.fem.Constant(V.mesh, 0.0), V.mesh
            )

        elif self.example_number == 201:  # todo make inhomogeneous
            self.u_exakt = TimeDependentUFLFunctor(
                lambda t: (
                    ufl.sin(ufl.pi * x[0])
                    * ufl.sin(ufl.pi * x[1])
                    * (ufl.cos(ufl.pi * t) + 4)
                ),
                V.mesh,
            )
            self.u_bc = self.u_exakt.copy()
            self.u0 = self.u_exakt.locked_evaluation()
            self.v_exakt = TimeDependentUFLFunctor(
                lambda t: (
                    -ufl.pi
                    * ufl.sin(ufl.pi * x[0])
                    * ufl.sin(ufl.pi * x[1])
                    * ufl.sin(ufl.pi * t)
                ),
                V.mesh,
            )
            self.v0 = self.v_exakt.locked_evaluation()
            self.f = TimeDependentUFLFunctor(
                lambda t: -ufl.pi
                * ufl.pi
                * ufl.sin(ufl.pi * x[0])
                * ufl.sin(ufl.pi * x[1])
                * ufl.cos(ufl.pi * t)
                + 2
                * ufl.pi
                * ufl.pi
                * ufl.sin(ufl.pi * x[0])
                * ufl.sin(ufl.pi * x[1])
                * (ufl.cos(ufl.pi * t) + 4),
                V.mesh,
            )
        elif self.example_number == 202:
            self.u_exakt = TimeDependentUFLFunctor(
                lambda t: (
                    t * x[0] * (1 - x[0]) * x[1] * (1 - x[1])
                    + (t**3 + 1) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
                ),
                V.mesh,
            )
            self.u_bc = self.u_exakt.copy()
            self.u0 = self.u_exakt.locked_evaluation()
            self.v_exakt = TimeDependentUFLFunctor(
                lambda t: (
                    x[0] * (1 - x[0]) * x[1] * (1 - x[1])
                    + (3 * t**2) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
                ),
                V.mesh,
            )
            self.v0 = self.v_exakt.locked_evaluation()
            self.f = TimeDependentUFLFunctor(
                lambda t: 6 * t * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
                + 2 * t * (x[1] * (1 - x[1]) + x[0] * (1 - x[0]))
                + 2
                * ufl.pi
                * ufl.pi
                * ufl.sin(ufl.pi * x[0])
                * ufl.sin(ufl.pi * x[1])
                * (t**3 + 1),
                V.mesh,
            )
        elif self.example_number == 204:
            base = ufl.sin(ufl.pi * x[0]) ** 2 * ufl.sin(ufl.pi * x[1]) ** 2
            laplace_base = (
                2
                * ufl.pi**2
                * (
                    ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) ** 2
                    + ufl.sin(ufl.pi * x[0]) ** 2 * ufl.cos(2 * ufl.pi * x[1])
                )
            )
            self.u_exakt = TimeDependentUFLFunctor(
                lambda t: base * ufl.exp(t), V.mesh
            )
            self.u_bc = self.u_exakt.copy()
            self.u0 = self.u_exakt.locked_evaluation()
            self.v_exakt = TimeDependentUFLFunctor(
                lambda t: base * ufl.exp(t), V.mesh
            )
            self.v0 = self.v_exakt.locked_evaluation()
            self.f = TimeDependentUFLFunctor(
                lambda t: (base - self.c**2 * laplace_base) * ufl.exp(t),
                V.mesh,
            )
        elif self.example_number == 223:
            mu, s, b, x0, x1 = 0.5, 0.2, 1.0, x[0], x[1]
            base_function = ufl.conditional(
                abs(x0 - mu) < s, ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 3, 0
            )
            base_function_derivative = ufl.conditional(
                abs(x0 - mu) < s,
                3
                * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 2
                * ufl.cos(ufl.pi * (x0 - (mu - s)) / (2 * s))
                * (ufl.pi / (2 * s)),
                0,
            )
            base_function_derivative2 = ufl.conditional(
                abs(x0 - mu) < s,
                6
                * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s))
                * ufl.cos(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 2
                * (ufl.pi / (2 * s)) ** 2
                - 3
                * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s)) ** 2
                * ufl.sin(ufl.pi * (x0 - (mu - s)) / (2 * s))
                * (ufl.pi / (2 * s)) ** 2,
                0,
            )
            sol1D, deriv1D = self.traveling_pulse_ufl(
                x0, base_function, base_function_derivative, mu, s, b
            )
            base_function_x1 = ufl.replace(base_function, {x0: x1})

            self.u_exakt = TimeDependentUFLFunctor(
                lambda t: sol1D(t) * base_function_x1
                + ufl.replace(sol1D(t), {x0: x1}) * base_function,
                V.mesh,
                reevaluation=True,
            )
            self.u0 = self.u_exakt.locked_evaluation()
            self.v_exakt = TimeDependentUFLFunctor(
                lambda t: deriv1D(t) * base_function_x1
                + ufl.replace(deriv1D(t), {x0: x1}) * base_function,
                V.mesh,
                reevaluation=True,
            )
            self.v0 = self.v_exakt.locked_evaluation()
            self.f = TimeDependentUFLFunctor(
                lambda t: -self.c_init**2
                * (
                    sol1D(t) * ufl.replace(base_function_derivative2, {x0: x1})
                    + ufl.replace(sol1D(t), {x0: x1}) * base_function_derivative2
                ),
                V.mesh,
                reevaluation=True,  # this is expensive
            )
        elif self.example_number == 228:
            mu, s, b, factor = 0.5, 0.4, 1.0, 0.75
            x0, x1 = x[0], x[1]

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
            sol1D, deriv1D = self.traveling_pulse_ufl(
                x0, base_function, base_function_derivative, mu, s, b
            )

            base_function_x1 = ufl.replace(base_function, {x0: x1})

            self.u_exakt = TimeDependentUFLFunctor(
                lambda t: sol1D(t) * base_function_x1
                + ufl.replace(sol1D(t), {x0: x1}) * base_function,
                V.mesh,
                reevaluation=True,
            )
            self.u0 = self.u_exakt.locked_evaluation()
            self.v_exakt = TimeDependentUFLFunctor(
                lambda t: deriv1D(t) * base_function_x1
                + ufl.replace(deriv1D(t), {x0: x1}) * base_function,
                V.mesh,
                reevaluation=True,
            )
            self.v0 = self.v_exakt.locked_evaluation()
            self.f = TimeDependentUFLFunctor(
                lambda t: -self.c**2
                * (
                    sol1D(t) * ufl.replace(base_function_derivative2, {x0: x1})
                    + ufl.replace(sol1D(t), {x0: x1}) * base_function_derivative2
                ),
                V.mesh,
                reevaluation=True,  # this is expensive
            )
        elif self.example_number == 230:
            # material parameter
            kappa_left = (
                self.c_init + 1.0
            )  # here the passed wave_propagation_speed is the jump
            kappa_right = 1.0
            mid = 0.5
            cell_idx = np.arange(0, V.dofmap.list.shape[0]).astype(np.int32)
            cell_midpoints_bums = cell_idx.copy().astype(np.int32)
            midpoints = dfx.mesh.compute_midpoints(
                V.mesh, V.mesh.geometry.dim, cell_midpoints_bums
            )

            cells_left = cell_idx[midpoints[:, 0] <= mid]
            cells_right = cell_idx[midpoints[:, 0] >= mid]

            kappa_left = dfx.fem.Expression(
                dfx.fem.Constant(V.mesh, PETSc.ScalarType(kappa_left)),
                V.element.interpolation_points,
            )
            kappa_right = dfx.fem.Expression(
                dfx.fem.Constant(V.mesh, PETSc.ScalarType(kappa_right)),
                V.element.interpolation_points,
            )

            kappa = dfx.fem.Function(V)
            kappa.interpolate(kappa_left, cells=cells_left)
            kappa.interpolate(kappa_right, cells=cells_right)
            self.c = kappa

            # definition of right hand side
            x_abscissa = 0.75
            x_std = 0.05
            y_abscissa = 0.5
            y_std = 0.05
            Amplitude = 1000
            omega = 4
            t_abscissa = 1
            t_std = 0.1

            self.f = TimeDependentUFLFunctor(
                lambda t: Amplitude
                * ufl.exp(-((x[0] - x_abscissa) ** 2 / (x_std**2)))
                * ufl.exp(-((x[1] - y_abscissa) ** 2 / (y_std**2)))
                * ufl.sin(omega * t)
                * ufl.exp(-((t - t_abscissa) ** 2 / (t_std**2))),
                V.mesh,
            )

            # zero initial data
            self.u0 = dfx.fem.Function(V)
            self.v0 = dfx.fem.Function(V)

            # zero Dirichlet bc and no exact solution
            self.u_bc = None
            self.u_exakt = None
            self.v_exakt = None
        else:
            raise NotImplementedError

    def traveling_pulse_ufl(
        self, x, base_function, base_function_derivative, mu=0.5, s=0.2, b=1.0
    ):
        import math

        right_support_bound = mu + s

        def sol(_t):
            phase = math.ceil((self.c_init * _t + right_support_bound) / b) % 2
            k = math.ceil((self.c_init * _t + right_support_bound) / b) // 2
            if phase:
                return ufl.replace(
                    base_function, {x: 2 * k * b + x - self.c_init * _t}
                ) - ufl.replace(base_function, {x: 2 * k * b - x - self.c_init * _t})
            else:
                return -ufl.replace(
                    base_function, {x: 2 * k * b - x - self.c_init * _t}
                ) + ufl.replace(
                    base_function, {x: 2 * (k - 1) * b + x - self.c_init * _t}
                )

        def derivative(_t):
            phase = math.ceil((self.c_init * _t + right_support_bound) / b) % 2
            k = math.ceil((self.c_init * _t + right_support_bound) / b) // 2
            if phase:
                return -self.c_init * (
                    ufl.replace(
                        base_function_derivative, {x: 2 * k * b + x - self.c_init * _t}
                    )
                    - ufl.replace(
                        base_function_derivative, {x: 2 * k * b - x - self.c_init * _t}
                    )
                )
            else:
                return -self.c * (
                    -ufl.replace(
                        base_function_derivative, {x: 2 * k * b - x - self.c_init * _t}
                    )
                    + ufl.replace(
                        base_function_derivative,
                        {x: 2 * (k - 1) * b + x - self.c_init * _t},
                    )
                )

        return sol, derivative

    def dfExpression(self, ufl_expr, V):
        if str(type(ufl_expr)) == "<class 'function'>":
            return ufl_expr
        elif ufl_expr.__nonzero__():
            return dfx.fem.Expression(ufl_expr, V.element.interpolation_points)
        else:
            return dfx.fem.Expression(
                dfx.fem.Constant(V.mesh, 0.0), V.element.interpolation_points
            )


if __name__ == "__main__":
    mesh = dfx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = dfx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    prob = ProblemData(9, V)
    u = dfx.fem.Function(V)
    f = prob.f
    F_tn = f.copy()
    f_tn = F_tn(0.0)
    print(F_tn, f_tn)
    F_tn.t.value = 1.0
    print(F_tn, f_tn)
    u.interpolate(prob.dfExpression(prob.u_exakt(1.0), V))
    print(u.x.array)
