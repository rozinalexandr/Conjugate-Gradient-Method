from typing import Callable

from methods.conjugate_gradients import ConjugateGradients
from methods.conjugate_gradients_1st_modification import ConjugateGradientsFirstModification
from methods.conjugate_gradients_2nd_modification import ConjugateGradientsSecondModification
from methods.conjugate_gradients_3rd_modification import ConjugateGradientsThirdModification
from mathematics.modification import get_alpha_k_single_factor_minimization, get_alpha_k_doubling_method
from mathematics.general import convert_y_values_to_plot
from drawing.plotter import make_plot


class Controller:
    def __init__(self, settings: dict):
        self._settings = settings

    @staticmethod
    def _get_minimization_method(method_name: str) -> Callable:
        """
        Using the Factory Method design pattern, we simplify the process of creating instances of minimization method
        classes. Thus, only those instances of minimization methods that are needed will be created.
        In order to add a new method, it is necessary to specify the method name as the key and the method class as
        the value in the methods_dict variable.

        :param method_name: String parameter with the name of the method.
        :return: Required class without initialization
        """

        methods_dict = {
            "Conjugate Gradients": ConjugateGradients,
            "Conjugate Gradients 1st Modification": ConjugateGradientsFirstModification,
            "Conjugate Gradients 2nd Modification": ConjugateGradientsSecondModification,
            "Conjugate Gradients 3rd Modification": ConjugateGradientsThirdModification
        }

        try:
            return methods_dict[method_name]
        except KeyError:
            raise KeyError(f"Wrong Method Name: {method_name}")

    def _get_alpha_k_calculating_method(self) -> Callable:
        """
        The method selects the required function to calculate alpha. It returns the function without calling it.
        Only one alpha calculation option can be selected, so if more or less calculation options are selected, an
        error will occur.

        :return: Required alpha calculating function without calling it.
        """

        methods_dict = {
            "Single-Factor Minimization": get_alpha_k_single_factor_minimization,
            "Doubling Method": get_alpha_k_doubling_method
        }

        selected_methods = [k for k, v in self._settings["Alpha k Selection"].items() if v]
        if len(selected_methods) != 1:
            raise Exception(f"Only one alpha calculating method could be selected. It has been selected "
                            f"{len(selected_methods)} methods")
        else:
            method_name = selected_methods[0]

        try:
            return methods_dict[method_name]
        except KeyError:
            raise KeyError(f"Wrong Method Name: {method_name}")

    def run(self) -> None:
        # select the necessary alpha calculation method without calling it. It will be passed to the calculator classes
        alpha_k_method = self._get_alpha_k_calculating_method()

        # selecting all the methods, which have 'True' value
        selected_methods = [k for k, v in self._settings["Methods Selection"].items() if v]
        if len(selected_methods) == 0:
            raise Exception("Please, select any minimization method")

        methods_result_list = []

        for selected_method_name in selected_methods:
            # initialize calculator class
            method = self._get_minimization_method(method_name=selected_method_name)
            method_instance = method(function=self._settings["Function Settings"]["Function"],
                                     x_0=self._settings["Function Settings"]["Starting Coordinates"],
                                     min_point=self._settings["Function Settings"]["Specified Minimum Coordinates"],
                                     accuracy=self._settings["Function Settings"]["Accuracy"],
                                     iteration_threshold=self._settings["Function Settings"]["Iteration Threshold"],
                                     alpha_k_calculating_method=alpha_k_method)

            result_list, function_value_at_known_point = method_instance.run_method()

            methods_result_list.append({"name": method_instance,
                                        "x": [i for i in range(len(result_list))],
                                        "y": convert_y_values_to_plot(result_list, function_value_at_known_point)})

        if self._settings["Plotter Settings"]["Plot"]:
            make_plot(methods_result_list)
