import numpy as np

from datetime import datetime
from typing import List, Callable, Tuple

from methods.abc_minimization_method import ABCMinimisationMethod
from mathematics.general import get_function_value_at_k_point
from mathematics.modification import get_x_next, get_h_k, get_beta_k, get_s_k_second_modification
from stopping_criteria.conjugate_stopping_criteria.conjugate_stopping_criteria import check_all_criteria


class ConjugateGradientsSecondModification(ABCMinimisationMethod):
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
        :param dimension: Number of unique variables in the function.
        :param free_symbols: List of unique sympy.Symbols, which are the function variables.
        """
        super().__init__(function, x_0, min_point, accuracy, iteration_threshold, alpha_k_calculating_method)

    def run_method(self) -> Tuple[np.ndarray[float | int], int | float]:
        """
        Main method of minimizer class. The full algorithm of the minimization method can be found here. See README for
        more information about the algorithm. This method also prints crucial information about the results of this
        modification.

        :return: tuple with np.ndarray of Function value at each iteration and Function value at known minimum point.
                 These data will be used to plot the graph.
        """
        start_time = datetime.now()
        function_value_at_known_min_point = get_function_value_at_k_point(function=self.function,
                                                                          free_symbols=self.free_symbols,
                                                                          x_current=self.min_point,
                                                                          dimension=self.dimension)

        iteration_counter = 0
        list_of_function_value_at_k_point = [get_function_value_at_k_point(function=self.function,
                                                                           free_symbols=self.free_symbols,
                                                                           x_current=self.x_0,
                                                                           dimension=self.dimension)]

        x_current = self.x_0
        x_previous = np.array([])
        s_previous = np.array([])
        h_current = np.eye(self.dimension)

        while True:
            if iteration_counter > self.iteration_threshold:
                print("!!!!!!!!! THRESHOLD !!!!!!!!!")
                self.print_result_info(min_point=x_current,
                                       iteration_number=iteration_counter,
                                       time=datetime.now() - start_time,
                                       reached_min_func_value=list_of_function_value_at_k_point[-1],
                                       started_func_value=list_of_function_value_at_k_point[0],
                                       known_min_function_value=function_value_at_known_min_point)
                break

            beta_current = get_beta_k(function=self.function,
                                      free_symbols=self.free_symbols,
                                      x_current=x_current,
                                      x_previous=x_previous,
                                      iteration_number=iteration_counter,
                                      dimension=self.dimension)

            s_current = get_s_k_second_modification(function=self.function,
                                                    free_symbols=self.free_symbols,
                                                    x_current=x_current,
                                                    beta_current=beta_current,
                                                    s_previous=s_previous,
                                                    h_current=h_current,
                                                    dimension=self.dimension,
                                                    iteration=iteration_counter)

            alpha_current = self.alpha_k_calculating_method(function=self.function,
                                                            free_symbols=self.free_symbols,
                                                            x_current=x_current,
                                                            s_current=s_current,
                                                            dimension=self.dimension)

            x_next = get_x_next(x_current=x_current,
                                alpha_current=alpha_current,
                                s_current=s_current)

            list_of_function_value_at_k_point.append(get_function_value_at_k_point(function=self.function,
                                                                                   free_symbols=self.free_symbols,
                                                                                   x_current=x_next,
                                                                                   dimension=self.dimension))
            iteration_counter += 1

            try:
                if check_all_criteria(function=self.function,
                                      free_symbols=self.free_symbols,
                                      x_current=x_next,
                                      x_previous=x_current,
                                      dimension=self.dimension,
                                      accuracy=self.accuracy):

                    self.print_result_info(min_point=x_next,
                                           iteration_number=iteration_counter,
                                           time=datetime.now() - start_time,
                                           reached_min_func_value=list_of_function_value_at_k_point[-1],
                                           started_func_value=list_of_function_value_at_k_point[0],
                                           known_min_function_value=function_value_at_known_min_point)
                    break

                else:
                    x_previous = x_current
                    x_current = x_next
                    s_previous = s_current
                    h_current = get_h_k(function=self.function,
                                        free_symbols=self.free_symbols,
                                        x_next=x_current,
                                        x_current=x_previous,
                                        dimension=self.dimension,
                                        h_previous=h_current)
            except TypeError:
                print("!!!!!!!!! ENTRAPMENT !!!!!!!!!")
                self.print_result_info(min_point=x_current,
                                       iteration_number=iteration_counter,
                                       time=datetime.now() - start_time,
                                       reached_min_func_value=list_of_function_value_at_k_point[-1],
                                       started_func_value=list_of_function_value_at_k_point[0],
                                       known_min_function_value=function_value_at_known_min_point)
                break

        return np.array(list_of_function_value_at_k_point), function_value_at_known_min_point
