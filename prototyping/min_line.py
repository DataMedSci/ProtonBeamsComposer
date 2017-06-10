from beprof.profile import Profile
import scipy.optimize
import numpy as np
from matplotlib import pyplot as plt


def chi(t, xc, yc):
    a, b = t
    temp = 0
    for i in range(len(xc)):
        temp += (yc[i] - (a * xc[i] + b))**2
    return temp


def vec(t, xc, yc):
    x, y = t
    return np.array([x, y])


xc1 = np.array([1, 2, 3.1, 4])
yc1 = np.array([1, 2.3, 2.9, 4])
xyc = (xc1, yc1)

x0 = np.array([1, 0])

# res = scipy.optimize.minimize(chi, x0, args=xyc, jac=vec, method='L-BFGS-B', options={"disp": True})

# print(res)


bp1 = Profile([(0, 0), (1, 0), (2, 0.2), (3, 0.5), (4, 1), (5, 0.2), (6, 0), (7, 0)])
bp2 = Profile([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0.5), (6, 1), (7, 0.2)])
bp3 = Profile([(0, 0.2), (1, 0.5), (2, 0.8), (3, 1), (4, 0.2), (5, 0), (6, 0), (7, 0)])
bp4 = Profile([(0, 0), (1, 0), (2, 0), (3, 0.2), (4, 0.4), (5, 0.6), (6, 1), (7, 0.2)])

sobp = Profile(np.array([bp1.x, bp1.y + bp2.y + bp3.y + bp4.y]).T)
bp_max_y = sobp.y.max()
sobp.y /= sobp.y.max()
norm_peaks = [bp1 / bp_max_y, bp2 / bp_max_y, bp3 / bp_max_y, bp4 / bp_max_y]


def calc_sobp_line(t, peaks):
    a, b = t
    tmp = 0
    for x in peaks.x:
        # tmp += (bp[x][1] - 1)**2
        tmp += (sobp.evaluate_at_x(x) - (a * x + b))**2
    return tmp

res = scipy.optimize.minimize(calc_sobp_line, x0, args=sobp, method='L-BFGS-B', options={"disp": True})
print(res['x'], res['fun'])

plt.plot(bp1.x, bp1.y)
plt.plot(bp2.x, bp2.y)
plt.plot(bp3.x, bp3.y)
plt.plot(bp4.x, bp4.y)

plt.plot(sobp.x, sobp.y, '--')

x_result = np.linspace(0, 7)
plt.plot(x_result, x_result*0.0577 + 0.34)

plt.show()
