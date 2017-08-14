import logging

import matplotlib.pyplot as plt
import numpy as np

from pbc.helpers import dump_data_to_file, load_data_from_dump

logger = logging.getLogger(__name__)


def plot_sobp(start, stop, sobp_object, target_modulation=None, target_range=None, step=0.01, helper_lines=True,
              save_plot=False, plot_path=None, display_plot=True, dump_data=False, file_path=''):
    """
    Plot SOBP from given starting point to stop point

    :param start: beginning point of the plot
    :param stop: ending point for the plot
    :param sobp_object: a SOBP class object used in plotting
    :param target_modulation: used as beginning of the plot
    :param target_range: used as end of the plot
    :param step: specifies how dense should be the plot
    :param helper_lines: turns on/off vertical and horizontal helper lines on the plot
    :param save_plot: if True - will attempt to save plot to disk
    :param plot_path: path with extension where the plot will be saved, ignored when save_plot is False
    :param display_plot: if True - displays a standard window with plot
    :param dump_data:
    :param file_path:
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

    if save_plot and plot_path:
        try:
            logger.info("Saving SOBP plot as {0}".format(plot_path))
            plt.savefig(plot_path)
        except ValueError as e:
            logger.error("Error occurred while saving SOBP plot!\n{0}".format(e))

    if file_path:
        try:
            dump_data_to_file(plot_domain, sobp_vals, file_name=file_path)
        except:
            logger.error("Invalid path given!")

    if display_plot:
        plt.show()

    plt.clf()


def plot_plateau(sobp_object, target_modulation, target_range, step=0.01, helper_lines=True, save_plot=False,
                 plot_path=None, display_plot=True, dump_path='', higher=True):
    """
    Plot SOBP plateau

    :param sobp_object: a SOBP class object used in plotting
    :param target_modulation: used as beginning of the plot
    :param target_range: used as end of the plot
    :param step: specifies how dense should be the plot
    :param helper_lines: turns on/off vertical and horizontal helper lines on the plot
    :param save_plot: if True - will attempt to save plot to disk
    :param plot_path: path with extension where the plot will be saved, ignored when save_plot is False
    :param display_plot: if True - displays a standard window with plot
    :param dump_path:
    :param higher: make bigger bounds on vertical axis
    """
    mod = sobp_object.modulation()
    ran = sobp_object.range()
    prox = sobp_object.proximal_range(val=0.990)
    beginning = target_range - target_modulation
    prox_val = sobp_object.y_at_x(beginning)
    ending = target_range

    plateau_domain = np.arange(beginning - 1.0, ending + 1.0, step)
    plateau = sobp_object.overall_sum(plateau_domain)
    plateau_factor = sobp_object._flat_plateau_factor_helper()

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
    plt.title("Modulation (99-90): {0:.3f}, Proximal ({4:.3f}): {1:.3f}, Distal (0.90): {2:.3f}, Plt-fac: {3:.3f}"
              .format(mod, prox, ran, plateau_factor, prox_val))

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

    # extract labels and create legend
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles, labels)

    if save_plot and plot_path:
        try:
            logger.info("Saving SOBP plateau plot as {0}".format(plot_path))
            plt.savefig(plot_path)
        except ValueError as e:
            logger.error("Error occurred while saving SOBP plot!\n{0}".format(e))

    if dump_path:
        try:
            dump_data_to_file(extended_plateau_domain, extended_plateau_vals, file_name=dump_path)
        except:
            logger.error("Invalid path given!")

    if display_plot:
        plt.show()

    plt.clf()


def make_plots_from_file(file_path, delimiter=';', plottype=None, save_path=None, second_file=None,
                         second_file_delimiter=';'):
    x_peak, y_peak = load_data_from_dump(file_path, delimiter)
    if second_file:
        x_peak2, y_peak2 = load_data_from_dump(second_file, second_file_delimiter)

    if plottype == "sobp":
        # todo: for now just return standard plot, another external file with
        # bp and positions and weights may be required
        plt.plot(x_peak, y_peak)
    elif plottype == "plateau":
        beginning = np.floor(x_peak[0])
        ending = x_peak[-1]

        # horizontal helper lines
        plt.plot([beginning - 1.0, ending + 1.0], [0.98, 0.98], color='orange')
        plt.plot([beginning - 1.0, ending + 1.0], [0.99, 0.99], color='green')
        plt.plot([beginning - 1.0, ending + 1.0], [1, 1], color='blue')
        plt.plot([beginning - 1.0, ending + 1.0], [1.01, 1.01], color='green')
        plt.plot([beginning - 1.0, ending + 1.0], [1.02, 1.02], color='orange')

        # main plot
        plt.plot(x_peak, y_peak, 'r', label='First plot')
        if second_file:
            plt.plot(x_peak2, y_peak2, 'b-', label='Second plot')

        axes = plt.gca()
        axes.set_xlim([beginning - 1.0, ending + 1.0])
        axes.set_ylim([0.97, 1.03])

        # set some denser labels on axis
        plt.xticks(np.arange(beginning - 1.0, ending + 1.1, 1))
        plt.yticks(np.arange(0.97, 1.03, 0.005))
    else:
        plt.plot(x_peak, y_peak, 'r-', label='First plot')
        if second_file:
            plt.plot(x_peak2, y_peak2, 'b-', label='Second plot')

    # plt.xlabel("Głębokośc w fantomie wodnym [mm]")
    # plt.ylabel("Relatywna dawka")

    axes = plt.gca()
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles, labels)

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.clf()
