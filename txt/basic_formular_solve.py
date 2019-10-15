from sympy import *
import numpy


def formular_solver():
    # 0. 关闭Latex语法
    init_printing(use_latex=False)

    # 1. 求根
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    aa = solve(Eq(y, x + x ** 2), x)
    print(aa)

    # 2. 解方程组 .lhs左边的代数式 .rhs右边的代数式
    # u_max, u_star, rho_max, rho_star, A, B = sympy.symbols('u_max u_star rho_max rho_star A B')
    u_max, u_star, rho_max, rho_star, A, B = symbols('u_max u_star rho_max rho_star A B')
    eq1 = Eq(0, u_max * rho_max * (1 - A * rho_max - B * rho_max ** 2))
    print(eq1)
    eq2 = Eq(0, u_max * (1 - 2 * A * rho_star - 3 * B * rho_star ** 2))
    eq3 = Eq(u_star, u_max * (1 - A * rho_star - B * rho_star ** 2))
    aa = eq2 - 3 * eq3
    print(aa)
    eq4 = Eq(eq2.lhs - 3 * eq3.lhs, eq2.rhs - 3 * eq3.rhs)
    print(eq4)
    # 方程简化
    print(eq4.simplify())
    # 函数将因式全部乘了起来，并且进行了化简
    print(eq4.expand())
    # 解方程
    rho_sol = solve(eq4, rho_star)[0]
    B_sol = solve(eq1, B)[0]
    # 带入求解
    A_sol = eq2.subs([(rho_star, rho_sol), (B, B_sol)])
    # 带入最终解
    A_sol[0].evalf(subs={u_star: 0.7, u_max: 1.0, rho_max: 10.0})

    # 3. 求微分：
    x, nu, t = symbols('x nu t')
    phi = exp(-(x - 4 * t) ** 2 / (4 * nu * (t + 1))) + exp(
        -(x - 4 * t - 2 * numpy.pi) ** 2 / (4 * nu * (t + 1)))
    phiprime = phi.diff(x)


if __name__ == '__main__':
    formular_solver()
    exit()
