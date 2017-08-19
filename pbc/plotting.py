import logging

import matplotlib.pyplot as plt
import numpy as np

from pbc.bragg_peak import BraggPeak
from pbc.helpers import dump_data_to_file, load_data_from_dump
from pbc.sobp import SOBP

logger = logging.getLogger(__name__)


def plot_sobp(start, stop, sobp_object, target_modulation=None, target_range=None, step=0.01, helper_lines=True,
              plot_path=None, display_plot=True, datafile_path=None, lang='en'):
    """
    Plot SOBP from given starting point to stop point

    :param start: beginning point of the plot
    :param stop: ending point for the plot
    :param sobp_object: a SOBP class object used in plotting
    :param target_modulation: used as beginning of the plot
    :param target_range: used as end of the plot
    :param step: specifies how dense should be the plot
    :param helper_lines: turns on/off vertical and horizontal helper lines on the plot
    :param plot_path: path with extension where the plot will be saved
    :param display_plot: if True - displays a standard window with plot
    :param datafile_path: path for saving datafile with plot data
    """
    mod = sobp_object.modulation()
    ran = sobp_object.range()
    plot_domain = np.arange(start - 1.0, stop + 1.0, step)
    sobp_vals = sobp_object.overall_sum(plot_domain)

    if helper_lines:
        plt.plot([start, stop], [1, 1], color='yellow')
        if target_modulation and target_range:
            begin = target_range - target_modulation
            plt.plot([begin, begin], [0, 1], color='yellow')
            plt.plot([target_range, target_range], [0, 1], color='yellow')
            if lang == 'pl':
                plt.title("Modulacja: {0:.3f}, Zasięg: {1:.3f}".format(mod, ran))
            else:
                plt.title("Modulation: {0:.3f}, Range: {1:.3f}".format(mod, ran))

    plt.plot(plot_domain, sobp_vals, color='red')

    # plot all peaks contained in SOBP
    for single_peak in sobp_object.component_peaks:
        tmp_vals = single_peak.evaluate(plot_domain)
        plt.plot(plot_domain, tmp_vals, color='black')
        # plot blue point on top of each peak
        plt.plot(plot_domain[tmp_vals.argmax()], tmp_vals.max(), 'bo')

    # limit axes and set some denser labels/ticks
    plt.xticks(np.arange(0, plot_domain[-1], 1))
    plt.yticks(np.arange(0, 1.11, 0.05))

    fig = plt.gcf()
    fig.set_size_inches(8, 6)

    axes = plt.gca()
    axes.set_xlim([plot_domain[0], plot_domain[-1]])
    axes.set_ylim([0, 1.1])

    if lang == 'pl':
        plt.xlabel("Głębokośc w fantomie wodnym [mm]")
        plt.ylabel("Relatywna dawka")
    else:
        plt.xlabel("Depth in water phantom [mm]")
        plt.ylabel("Relative dose")

    if plot_path:
        try:
            logger.info("Saving SOBP plot as {0}".format(plot_path))
            plt.savefig(plot_path)
        except ValueError as e:
            logger.error("Error occurred while saving SOBP plot!\n{0}".format(e))

    if datafile_path:
        try:
            dump_data_to_file(plot_domain, sobp_vals, file_name=datafile_path)
        except IOError:
            logger.error("Invalid path for datafile given!")

    if display_plot:
        plt.show()

    plt.clf()


def plot_plateau(sobp_object, target_modulation, target_range, step=0.01, helper_lines=True,
                 plot_path=None, display_plot=True, datafile_path=None, higher=True, lang='en'):
    """
    Plot SOBP plateau

    :param sobp_object: a SOBP class object used in plotting
    :param target_modulation: used as beginning of the plot
    :param target_range: used as end of the plot
    :param step: specifies how dense should be the plot
    :param helper_lines: turns on/off vertical and horizontal helper lines on the plot
    :param plot_path: path with extension where the plot will be saved
    :param display_plot: if True - displays a standard window with plot
    :param datafile_path:
    :param higher: make bigger bounds on vertical axis
    """
    mod = sobp_object.modulation()
    ran = sobp_object.range()
    prox = sobp_object.proximal_range(val=0.990)
    prox_val = sobp_object.y_at_x(prox)
    beginning = target_range - target_modulation
    ending = target_range

    plateau_domain = np.arange(beginning - 1.0, ending + 1.0, step)
    plateau = sobp_object.overall_sum(plateau_domain)

    extended_plateau_domain = np.arange(beginning - 1.0, ending + 1.0, step)
    extended_plateau_vals = sobp_object.overall_sum(extended_plateau_domain)

    if helper_lines:
        # horizontal helper lines
        plt.plot([beginning - 1.0, ending + 1.0], [0.98, 0.98], color='orange')
        plt.plot([beginning - 1.0, ending + 1.0], [0.99, 0.99], color='green')
        plt.plot([beginning - 1.0, ending + 1.0], [1, 1], color='blue')
        plt.plot([beginning - 1.0, ending + 1.0], [1.01, 1.01], color='green')
        plt.plot([beginning - 1.0, ending + 1.0], [1.02, 1.02], color='orange')
        # vertical helper lines
        plt.plot([beginning, beginning], [0.88, 1.025], color='red', label='start = %s' % beginning)
        plt.plot([ending, ending], [0.88, 1.025], color='magenta', label='end = %s' % ending)
        # 99-90 points
        plt.plot(prox, prox_val, 'ro', label='proximal')
        plt.plot(ran, 0.90, 'co', label='distal')

    # result plateau
    plt.plot(plateau_domain, plateau, label='SOBP', color='black')
    if lang == 'pl':
        plt.title("Modulacja (99-90): {0:.3f}, Proximal ({3:.3f}): {1:.3f}, Distal (0.90): {2:.3f}"
                  .format(mod, prox, ran, prox_val))
    else:
        plt.title("Modulation (99-90): {0:.3f}, Proximal ({3:.3f}): {1:.3f}, Distal (0.90): {2:.3f}"
                  .format(mod, prox, ran, prox_val))

    fig = plt.gcf()
    fig.set_size_inches(8, 6)

    # limit axes and set some denser labels/ticks
    axes = plt.gca()
    axes.set_xlim([np.floor(beginning - 1.0), ending + 1.0])
    plt.xticks(np.arange(np.floor(beginning - 1.0), ending + 1.1, 1))
    if higher:
        axes.set_ylim([0.88, 1.025])
        plt.yticks(np.arange(0.88, 1.026, 0.005))
    else:
        axes.set_ylim([0.9875, 1.0125])
        plt.yticks(np.arange(0.9875, 1.0126, 0.0025))

    if lang == 'pl':
        plt.xlabel("Głębokośc w fantomie wodnym [mm]")
        plt.ylabel("Relatywna dawka")
    else:
        plt.xlabel("Depth in water phantom [mm]")
        plt.ylabel("Relative dose")

    # extract labels and create legend
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles, labels)

    if plot_path:
        try:
            logger.info("Saving SOBP plateau plot as {0}".format(plot_path))
            plt.savefig(plot_path)
        except ValueError as e:
            logger.error("Error occurred while saving SOBP plot!\n{0}".format(e))

    if datafile_path:
        try:
            dump_data_to_file(extended_plateau_domain, extended_plateau_vals, file_name=datafile_path)
        except IOError:
            logger.error("Invalid path given!")

    if display_plot:
        plt.show()

    plt.clf()


def make_plots_from_file(file_path, delimiter=";", plottype=None, save_path=None, second_file=None,
                         second_file_delimiter=";", zoom_on_plateau=False):

    x_peak, y_peak = load_data_from_dump(file_path, delimiter)

    if plottype == "sobp" or plottype == "plateau":
        if not second_file:
            logger.error("Second file with positions and weights is required to plot SOBP!")
            return -1
        else:
            x_peak2, y_peak2 = load_data_from_dump(second_file, second_file_delimiter)

        bp_object = BraggPeak(x_peak, y_peak)

        sobp_params = []
        for idx, val in enumerate(x_peak2):
            sobp_params.append([val, y_peak2[idx]])

        sobp_object = SOBP(bragg_peaks=bp_object, param_list=sobp_params)
        sobp_object.def_domain = np.arange(x_peak2.min() - 1, x_peak2.max() + 1, 0.01)

        if plottype == "sobp":
            plot_sobp(x_peak2[0], x_peak2[-1], sobp_object, plot_path=save_path)
            return
        else:
            rng = x_peak2[-1]
            mdl = x_peak2[-1] - x_peak2[0]
            plot_plateau(sobp_object=sobp_object, target_range=rng, target_modulation=mdl, plot_path=save_path,
                         higher=not zoom_on_plateau)
            return
    else:
        plt.plot(x_peak, y_peak, 'r-', label='First plot')
        if second_file:
            x_peak2, y_peak2 = load_data_from_dump(second_file, second_file_delimiter)
            plt.plot(x_peak2, y_peak2, 'b-', label='Second plot')

        if save_path:
            plt.savefig(save_path)

        plt.show()
        plt.clf()
