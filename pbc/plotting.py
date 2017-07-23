import matplotlib.pyplot as plt
import numpy as np


def plot_sobp(start, stop, sobp_object):
    sobp_vals = sobp_object.overall_sum()
    mod = sobp_object.modulation()
    ran = sobp_object.range()
    begin = ran - mod

    plot_domain = np.arange(start, stop, 0.01)

    plt.plot([start, stop], [1, 1], color='yellow')
    plt.plot([begin, begin])
    plt.plot(plot_domain, sobp_vals, label='sum', color='red')
    plt.title("Modulation: {0:.3f}, Range: {1:.3f}".format(mod, ran))
    plt.show()


def plot_plateau(start, stop, sobp_object, target_modulation):
    """Plot plateau only"""
    mod = sobp_object.modulation()
    ran = sobp_object.range()
    plateau_domain = np.arange(0, stop, 0.1)
    plateau = sobp_object.overall_sum(plateau_domain)
    plateau_factor = sum([abs(pp - 1.0) for pp in plateau])

    plt.plot([start, stop], [0.98, 0.98], color='orange')
    plt.plot([start, stop], [0.99, 0.99], color='green')
    plt.plot([start, stop], [1, 1], color='blue')
    plt.plot([start, stop], [1.02, 1.02], color='orange')
    plt.plot([start, stop], [1.01, 1.01], color='green')
    plt.plot(plateau_domain, plateau, label='sum', color='red')
    plt.title("Modulation: {0:.3f}, Range: {1:.3f}, Mod-diff: {2:.2f}, Plateau-factor: {3:.4f}"
              .format(mod, ran, abs(mod - target_modulation), plateau_factor))
    axes = plt.gca()
    axes.set_xlim([0.0, ran + 1.0])
    axes.set_ylim([0.95, 1.05])
    plt.show()
