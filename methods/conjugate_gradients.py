from datetime import datetime
from typing import List, Callable

from methods.abc_minimization_method import ABCMinimisationMethod
from mathematics.general import get_function_value_at_k_point
from mathematics.modification import get_s_0, get_x_next, get_beta_k, get_s_k_base_modification
from stopping_criteria.conjugate_stopping_criteria.conjugate_stopping_criteria import check_all_criteria


class ConjugateGradients(ABCMinimisationMethod):
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

    def run_method(self):
        start_time = datetime.now()
        iteration_counter = 0
        list_of_function_value_at_k_point = [get_function_value_at_k_point(function=self.function,
                                                                           free_symbols=self.free_symbols,
                                                                           x_current=self.x_0,
                                                                           dimension=self.dimension)]

        s_0 = get_s_0(function=self.function,
                      free_symbols=self.free_symbols,
                      x_0=self.x_0,
                      dimension=self.dimension)

        alpha_0 = self.alpha_k_calculating_method(function=self.function,
                                                  free_symbols=self.free_symbols,
                                                  x_current=self.x_0,
                                                  s_current=s_0,
                                                  dimension=self.dimension)

        x_1 = get_x_next(x_current=self.x_0,
                         alpha_current=alpha_0,
                         s_current=s_0)

        list_of_function_value_at_k_point.append(get_function_value_at_k_point(function=self.function,
                                                                               free_symbols=self.free_symbols,
                                                                               x_current=x_1,
                                                                               dimension=self.dimension))

        if check_all_criteria(function=self.function,
                              free_symbols=self.free_symbols,
                              x_current=x_1,
                              x_previous=self.x_0,
                              dimension=self.dimension,
                              accuracy=self.accuracy):
            print(f"Done, x = {x_1}")

        else:
            x_previous = self.x_0
            x_current = x_1
            s_previous = s_0

            while True:
                iteration_counter += 1
                beta_current = get_beta_k(function=self.function,
                                          free_symbols=self.free_symbols,
                                          x_current=x_current,
                                          x_previous=x_previous,
                                          iteration_number=iteration_counter,
                                          dimension=self.dimension)

                s_current = get_s_k_base_modification(function=self.function,
                                                      free_symbols=self.free_symbols,
                                                      x_current=x_current,
                                                      beta_current=beta_current,
                                                      s_previous=s_previous,
                                                      dimension=self.dimension)

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

                if iteration_counter > self.iteration_threshold:
                    print("THRESHOLD")
                    break

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
                                           started_func_value=list_of_function_value_at_k_point[0])
                    break
                else:
                    x_previous = x_current
                    x_current = x_next
                    s_previous = s_current

        return list_of_function_value_at_k_point
