from abc import ABC, abstractmethod

import numpy as np
from sympy import sympify

import re
import datetime
from typing import List, Callable


class ABCMinimisationMethod(ABC):
    def __init__(self,
                 function: str,
                 x_0: List[int | float],
                 min_point: List[int | float],
                 accuracy: int,
                 iteration_threshold: int,
                 alpha_k_calculating_method: Callable):
        """
        All the minimization methods shares the same input data.

        :param function: Function in a string format.
        :param x_0: List with starting coordinates.
        :param min_point: List with known minimum coordinates.
        :param accuracy: Given accuracy, with which we need to find the points of minimum of the function.
        :param iteration_threshold: Number of iterations after which the minimization process terminates.
        :param alpha_k_calculating_method: Function reference for alpha calculation.
        :param _dimension: Number of unique variables in the function.
        :param _free_symbols: List of unique sympy.Symbols, which are the function variables.
        """
        self.function = sympify(function)
        self.x_0 = np.array(x_0)
        self.min_point = np.array(min_point)
        self.accuracy = 10 ** accuracy
        self.iteration_threshold = iteration_threshold
        self.alpha_k_calculating_method = alpha_k_calculating_method
        self.dimension = len(x_0)
        self.free_symbols = sorted(self.function.free_symbols, key=lambda sym: sym.name)

    def __str__(self):
        """
        For better representation and further logging the name of the method is the name of the corresponding class.
        All inherited classes must be named in the CamelCase style. This ensures that, when printed, the class name
        will be separated by capitalization using spaces.
        E.g. the name of a class is 'MyNewClass' and when we print it, we will get 'My New Class'

        :return: String with a name of a class, separated by spaces.
        """
        return " ".join(re.findall('[A-Z][^A-Z]*', type(self).__name__))

    @staticmethod
    def euclidean_distance(p: np.ndarray[float | int],
                           q: np.ndarray[float | int]) -> str:
        """
        Method calculates the Euclidian Distance between 2 points.

        :param p: First point.
        :param q: Second point.

        :return: string with Euclidian Distance.
        """

        p = np.array(p).astype(np.float32)
        q = np.array(q).astype(np.float32)

        return f"{np.sqrt(np.sum((p - q) ** 2)):.2e}"

    @staticmethod
    def reformat_min_coordinates_output(min_point: np.ndarray[float | int],
                                        precision: int) -> str:
        """
        Method changes minimum point coordinates, so they could be displayed in a better way.

        :param min_point: numpy array with minimum point coordinates.
        :param precision: number of numbers to be displayed after decimal point.

        :return: string with reformatted coordinates.
        """

        return "(" + ", ".join([f"{i:.{precision + 1}}" for i in min_point]) + ")"

    def print_result_info(self,
                          min_point: np.ndarray[float | int],
                          iteration_number: int,
                          time: datetime.timedelta,
                          reached_min_func_value: float | int,
                          started_func_value: float | int,
                          known_min_function_value: float | int,
                          additional_message: str = "") -> None:
        """
        Method prints all the necessary information of minimization process.

        :param min_point: Reached point of minimum
        :param iteration_number: Number of iterations.
        :param time: Taken time for minimization.
        :param reached_min_func_value: Function value at reached point of minimum.
        :param started_func_value: Function value at starting point.
        :param known_min_function_value: Function value at known minimum point.
        :param additional_message: Optional string to be written at the very beginning, before the delimiter.
        """

        delimiter_str = "=" * 70 + "\n"
        method_name_str = f"Method name: {self}\n"
        resulting_min_str = f"The resulting point of minimum: {self.reformat_min_coordinates_output(min_point, 8)}\n"
        target_min_str = f"Target point of minimum: {'(' + ', '.join(self.min_point.astype(str)) + ')'}\n"
        iterations_str = f"Number of Iterations: {iteration_number}\n"
        time_str = f"Execution time: {time}\n"
        reached_min_func_value_str = f"Function value at the reached minimum point: " \
                                     f"{float(reached_min_func_value):.2e}\n"
        started_func_value_str = f"Function value at starting point: {started_func_value}\n"
        known_min_function_value_str = f"Function value at known minimum point (Phi*): {known_min_function_value}\n"
        euclidian_distance_str = "Euclidian distance between Target and Resulting minimum points (Delta X): " \
                                 + self.euclidean_distance(self.min_point, min_point) + "\n"
        delta_phi_str = f"Difference between function values at known and reached minimum points (Delta Phi): " \
                        f"{abs(float(known_min_function_value) - float(reached_min_func_value)):.2e}\n"

        print(additional_message
              + delimiter_str
              + method_name_str
              + resulting_min_str
              + target_min_str
              + euclidian_distance_str
              + iterations_str
              + time_str
              + started_func_value_str
              + known_min_function_value_str
              + reached_min_func_value_str
              + delta_phi_str
              + delimiter_str)

    @abstractmethod
    def run_method(self):
        pass
