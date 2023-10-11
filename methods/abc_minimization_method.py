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

        delimiter = "=" * 70 + "\n"
        method_name = f"Method name: {self}\n"
        resulting_min = f"The resulting point of minimum: {min_point}\n"
        target_min = f"Target point of minimum: {self.min_point}\n"
        iterations = f"Number of Iterations: {iteration_number}\n"
        time = f"Execution time: {time}\n"
        reached_min_func_value = f"Function value at the reached minimum point: {reached_min_func_value}\n"
        started_func_value = f"Function value at starting point: {started_func_value}\n"
        known_min_function_value = f"Function value at known minimum point: {known_min_function_value}\n"

        print(additional_message + delimiter + method_name + resulting_min + target_min + iterations + time
              + known_min_function_value + started_func_value + reached_min_func_value + delimiter)

    @abstractmethod
    def run_method(self):
        pass
