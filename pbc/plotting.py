import matplotlib.pyplot as plt
import numpy as np


def plot_sobp(start, stop, sobp_object, target_modulation=None, target_range=None, step=0.01, helper_lines=True):
    """Plot whole SOBP from given start to stop"""
    mod = sobp_object.modulation()
    ran = sobp_object.range()
    plot_domain = np.arange(start, stop, step)
    sobp_vals = sobp_object.overall_sum(plot_domain)

    if helper_lines:
        plt.plot([start, stop], [1, 1], color='yellow')
        if target_modulation and target_range:
            begin = target_range - target_modulation
            plt.plot([begin, begin], [0, 1], color='yellow')
            plt.plot([target_range, target_range], [0, 1], color='yellow')
    plt.plot(plot_domain, sobp_vals, color='red')
    plt.title("Modulation: {0:.3f}, Range: {1:.3f}".format(mod, ran))

    plt.show()


def plot_plateau(sobp_object, target_modulation, target_range, step=0.01):
    """
    Plot plateau only with some helper lines.

    todo: add save to disk option

    :param sobp_object:
    :param target_modulation:
    :param target_range:
    :param step:
    """
    mod = sobp_object.modulation()
    ran = sobp_object.range()
    beginning = target_range - target_modulation
    ending = target_range
    plateau_domain = np.arange(beginning, ending, step)
    plateau = sobp_object.overall_sum(plateau_domain)
    plateau_factor = sum([abs(pp - 1.0) for pp in plateau])

    extended_plateau_domain = np.arange(beginning - 1.0, ending + 1.0, step)
    extended_plateau_vals = sobp_object.overall_sum(extended_plateau_domain)

    # horizontal helper lines
    plt.plot([beginning - 1.0, ending + 1.0], [0.98, 0.98], color='orange')
    plt.plot([beginning - 1.0, ending + 1.0], [0.99, 0.99], color='green')
    plt.plot([beginning - 1.0, ending + 1.0], [1, 1], color='blue')
    plt.plot([beginning - 1.0, ending + 1.0], [1.02, 1.02], color='orange')
    plt.plot([beginning - 1.0, ending + 1.0], [1.01, 1.01], color='green')
    # vertical helper lines
    plt.plot([beginning, beginning], [0.94, 1.04], color='purple', label='begin = %s' % beginning)
    plt.plot([ending, ending], [0.96, 1.04], color='blue', label='end = %s' % ending)
    # result plateau
    plt.plot(extended_plateau_domain, extended_plateau_vals, label='extended plateau', color='red')
    plt.plot(plateau_domain, plateau, label='sum', color='black')
    plt.title("Modulation: {0:.3f}, Range: {1:.3f}, Mod-diff: {2:.2f}, Plateau-factor: {3:.4f}"
              .format(mod, ran, abs(mod - target_modulation), plateau_factor))

    # limit axes
    axes = plt.gca()
    axes.set_xlim([beginning - 1.0, ending + 1.0])
    axes.set_ylim([0.96, 1.04])
    # extract labels and create legend
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles, labels)

    # set some denser labels on axis
    plt.xticks(np.arange(beginning - 1.0, ending + 1.1, 1))
    plt.yticks(np.arange(0.96, 1.04, 0.01))

    plt.show()
