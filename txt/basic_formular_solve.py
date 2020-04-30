from sympy import *
import numpy


def formular_solver():
    # 0. 关闭Latex语法
    init_printing(use_latex=False)
    s = "1 + 2"
    eval(s)
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

    """
      x+y+z-2=0
      2x-y+z+1=0
      x+2y+2z-3=0
    """
    from sympy import *
    x, y, z = symbols("x y z")
    # 默认等式为0的形式
    print("======默认等式为0的形式 =======")
    eq = [x + y + z - 2, 2 * x - y + z + 1, x + 2 * y + 2 * z - 3]
    result = linsolve(eq, [x, y, z])
    print(result)
    print(latex(result))
    # 矩阵形式
    print("======矩阵形式 =======")
    eq = Matrix(([1, 1, 1, 2], [2, -1, 1, -1], [1, 2, 2, 3]))
    result = linsolve(eq, [x, y, z])
    print(result)
    print(latex(result))
    # 增广矩阵形式
    print("======增广矩阵形式 =======")
    A = Matrix([[1, 1, 1], [2, -1, 1], [1, 2, 2]])
    b = Matrix([[2], [-1], [3]])
    system = A, b
    result = linsolve(system, x, y, z)
    print(result)
    print(latex(result))

    """
      x**2+y**2-2=0
      x**3+y**3=0
    """
    import sympy as sy
    x, y = sy.symbols("x y")
    eq = [x ** 2 + y ** 3 - 2, x ** 3 + y ** 3]
    result = sy.nonlinsolve(eq, [x, y])
    print(result)
    print(sy.latex(result))

    # 3. 求微分：
    x, nu, t = symbols('x nu t')
    phi = exp(-(x - 4 * t) ** 2 / (4 * nu * (t + 1))) + exp(
        -(x - 4 * t - 2 * numpy.pi) ** 2 / (4 * nu * (t + 1)))
    phiprime = phi.diff(x)

    from sympy import *
    # 初始化
    x = symbols('x')
    f = symbols('f', cls=Function)
    # 表达式
    expr1 = Eq(f(x).diff(x, x) - 2*f(x).diff(x) + f(x), sin(x))
    # 求解微分方程
    r1 = dsolve(expr1, f(x))
    print(r1)
    print("原式：", latex(expr1))
    print("求解后：", latex(r1))

if __name__ == '__main__':
    formular_solver()
    exit()
