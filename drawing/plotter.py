import matplotlib.pyplot as plt
from itertools import cycle, islice
from typing import List, Dict


def get_styles(methods_number: int) -> list:
    """
    Function accepts as an input parameter the number of methods to be displayed on the plot. Knowing this number,
    the function repeats the available styles in a circle. Thus, the output is a list of available styles for the graph,
    the same length as the number of methods to be plotted.

    :param methods_number: The number of input methods to be plotted.
    :return: List of available styles that are repeated a given number of times.
    """
    available_styles = ["-", "--", "-.", ":"]
    return list(islice(cycle(available_styles), 0, methods_number + 1))


def make_plot(list_of_results: List[Dict]) -> None:
    """
    Function plots dependency graphs for the selected methods. It accepts an unlimited number of dictionaries with all
    necessary information for graphing.

    The input dictionaries must be as follows:
    {"name": "Method Name", "x": Dict[int], "y": Dict[float | int]}
    """
    fig, ax = plt.subplots()

    list_of_styles = get_styles(len(list_of_results))

    for index, method_dict in enumerate(list_of_results):
        x, y = method_dict["x"], method_dict["y"]
        ax.plot(x, y, linestyle=list_of_styles[index], label=method_dict["name"])

    ax.set_title("Comparing modifications")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("log |f(x) - f(x*)|")
    ax.legend()

    plt.show()
