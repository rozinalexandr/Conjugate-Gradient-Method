import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


class Plotter:
    def __init__(self, function_value_at_minimum_point):
        self._function_value_at_minimum_point = np.array(function_value_at_minimum_point).astype(np.float64)

    def _prepare_y_values(self, values):
        y = []
        for i in np.array(values).astype(np.float64):
            y.append(np.log10(abs(i - self._function_value_at_minimum_point)))
        return y

    def plot(self, base_method_values, first_modification_values, second_modification_values, third_modification_values):
        x1 = [i for i in range(len(base_method_values))]
        y1 = self._prepare_y_values(base_method_values)

        x2 = [i for i in range(len(first_modification_values))]
        y2 = self._prepare_y_values(first_modification_values)

        x3 = [i for i in range(len(second_modification_values))]
        y3 = self._prepare_y_values(second_modification_values)

        x4 = [i for i in range(len(third_modification_values))]
        y4 = self._prepare_y_values(third_modification_values)

        fig = plt.figure()

        ax = fig.add_subplot()

        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

        plt.title("Comparing modifications")
        plt.xlabel("Number of iterations")
        plt.ylabel("log |f(x) - f(x*)|")

        ax.plot(x1, y1, "-", label="Base method", color="tab:blue", )
        ax.plot(x2, y2, "--", label="First modification", color="tab:red")
        ax.plot(x3, y3, "-.", label="Second modification", color="tab:orange")
        ax.plot(x4, y4, ":", label="Third modification", color="tab:green")

        plt.legend()

        plt.show()
