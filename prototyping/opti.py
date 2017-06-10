import scipy.optimize
import numpy as np


def function_one(t):
    x, y = t
    return x ** 2 + y ** 2


def vec(t):
    x, y = t
    return np.array([2 * x, 2 * y])

#########################


def f2(t):
    x, y = t
    return (x-1)**2 + 3 * (y-2)**4 + 5


def vec2(t):
    x, y = t
    return np.array([2 * x - 2, 6 * y - 12])


def rosen(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


x0 = np.array([1.3, 0.7])
x01 = np.array([1.3, 0.7, 0.8, 1.9])
x1 = np.array([1, 2])
ar = np.array([1.])

# res = scipy.optimize.fmin_l_bfgs_b(func=function_one, x0=x0, fprime=vec, args=ar)

res = scipy.optimize.minimize(f2, x0, method='L-BFGS-B', jac=vec2, options={"disp": True})
# print(res)
# print("Result: ", res.get('x'))
