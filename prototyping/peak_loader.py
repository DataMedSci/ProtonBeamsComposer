import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_funcition_from_points(x, y):
    # todo: this has to be improved, especially end values...
    z = np.polyfit(x, y, 5)
    f = np.poly1d(z)
    return f


if __name__ == '__main__':
    # todo: use argparse
    if len(sys.argv) != 3:
        print("Wrong invocation!")
        print("Usage: %s BP_FILE POSITIONS_FILE" % sys.argv[0])
        exit(1)

    # load bp and get 2 first columns
    with open(sys.argv[1], 'r') as bp_file:
        data = pd.read_csv(bp_file, sep=';')

    print("Loaded columns:\n%s\nfrom file: %s\n" % (data.columns, sys.argv[1]))

    # get first and second col and change representation to numpy-like
    distance = data[data.columns[0]].as_matrix()
    dose = data[data.columns[1]].as_matrix()

    peak_max_position = distance[dose.argmax()]
    print("Dose == 1 is at range %s" % peak_max_position)

    # load file with positions and weights
    with open(sys.argv[2]) as pos_file:
        pos_we_data = pd.read_csv(pos_file, sep=';')

    print("\nLoaded pos/weights from file: %s\n" % sys.argv[2])

    positions = pos_we_data['position'].as_matrix()
    weights = pos_we_data['weight'].as_matrix()

    single_peak_plots = []
    for i in range(len(positions)):
        print("Peak[%s] -> position: %s with weight: %s" % (i, positions[i], weights[i]))

        # move peak to desired location if needed
        if abs(positions[i] - peak_max_position) > 0.01:
            print("Shifting peak by: %s" % abs(positions[i] - peak_max_position))
            temp_pos = distance + (positions[i] - peak_max_position)
            temp_dose = dose * weights[i]
            print(temp_pos)
            print(temp_dose)
            single_peak_plots.append(
                plt.plot(temp_pos, temp_dose, 'r', label="Pos " + str(positions[i]) + " wei: " + str(weights[i]))[0]
            )

    # plt.plot(distance, dose)
    # plt.title("Pristine peak (rs0)")

    plt.legend(handles=single_peak_plots)
    plt.show()

    peak_f = make_funcition_from_points(distance, dose)

    peak_f_list = []
    # construct polymonial for each peak using its position and weight
    for j in range(len(positions)):
        peak_f_list.append(
            make_funcition_from_points(
                x=distance + (positions[j] - peak_max_position),
                y=dose * weights[j])
        )

    domain = np.arange(0, 25, 0.01)
    # plot calculated peak functions
    for f in peak_f_list:
        plt.plot(domain, f(domain))

    # sum peak polynomials to one
    for p in range(len(peak_f_list)-1):
        sobp_poly = np.polyadd(peak_f_list[p], peak_f_list[p+1])

    plt.plot(domain, sobp_poly(domain), 'b+')
    plt.show()

    plt.plot(domain, peak_f(domain))
    plt.show()
