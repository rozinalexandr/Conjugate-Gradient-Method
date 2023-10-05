import numpy as np
import sympy

from typing import List

from mathematics.general import get_symbol_value_mapping, get_derivative, get_norm_of_vector


def check_first_stopping_criteria(function: sympy.core.add.Add,
                                  free_symbols: List[sympy.Symbol],
                                  x_current: np.ndarray[float | int],
                                  x_previous: np.ndarray[float | int],
                                  dimension: int,
                                  accuracy: float) -> bool:
    """
    Function is checking the First Stopping Criteria. For more information about Stopping Criteria check README file.

    :param function: Origin function which was transformed with sympy.sympify().
    :param free_symbols: List of unique sympy.Symbols, which are the function variables.
    :param x_current: numpy.ndarray with current function solutions.
    :param x_previous: numpy.ndarray with previous function solutions.
    :param dimension: Number of unique variables in the function.
    :param accuracy: Given accuracy, with which we need to find the points of minimum of the function.

    :return: Boolean value whether the current solution fulfills the criteria.
    """

    previous_symbol_value_mapping = get_symbol_value_mapping(free_symbols, x_previous, dimension)
    current_symbol_value_mapping = get_symbol_value_mapping(free_symbols, x_current, dimension)

    previous_function_solution = function.subs(previous_symbol_value_mapping)
    current_function_solution = function.subs(current_symbol_value_mapping)

    return previous_function_solution - current_function_solution < accuracy * (1 + abs(current_function_solution))


def check_second_stopping_criteria(x_current: np.ndarray[float | int],
                                   x_previous: np.ndarray[float | int],
                                   accuracy: float) -> bool:
    """
    Function is checking the Second Stopping Criteria. For more information about Stopping Criteria check README file.

    :param x_current: numpy.ndarray with current function solutions.
    :param x_previous: numpy.ndarray with previous function solutions.
    :param accuracy: Given accuracy, with which we need to find the points of minimum of the function.

    :return: Boolean value whether the current solution fulfills the criteria.
    """

    return get_norm_of_vector(x_previous - x_current) < accuracy ** 0.5 * (1 + get_norm_of_vector(x_current))


def check_third_stopping_criteria(function: sympy.core.add.Add,
                                  free_symbols: List[sympy.Symbol],
                                  x_current: np.ndarray[float | int],
                                  dimension: int,
                                  accuracy: float) -> bool:
    """
    Function is checking the Third Stopping Criteria. For more information about Stopping Criteria check README file.

    :param function: Origin function which was transformed with sympy.sympify().
    :param free_symbols: List of unique sympy.Symbols, which are the function variables.
    :param x_current: numpy.ndarray with current function solutions.
    :param dimension: Number of unique variables in the function.
    :param accuracy: Given accuracy, with which we need to find the points of minimum of the function.

    :return: Boolean value whether the current solution fulfills the criteria.
    """

    symbol_value_mapping = get_symbol_value_mapping(free_symbols, x_current, dimension)

    function_derivative_with_substitution = []
    for i in range(dimension):
        function_derivative_with_substitution.append(get_derivative(function, free_symbols[i]).
                                                     subs(symbol_value_mapping))

    current_function_solution = function.subs(symbol_value_mapping)

    return (get_norm_of_vector(np.array(function_derivative_with_substitution))
            <= accuracy ** (1 / 3) * (1 + abs(current_function_solution)))


def check_all_criteria(function: sympy.core.add.Add,
                       free_symbols: List[sympy.Symbol],
                       x_current: np.ndarray[float | int],
                       x_previous: np.ndarray[float | int],
                       dimension: int,
                       accuracy: float) -> bool:
    """
    Function is checking all the Stopping Criteria. For more information about Stopping Criteria check README file.

    :param function: Origin function which was transformed with sympy.sympify().
    :param free_symbols: List of unique sympy.Symbols, which are the function variables.
    :param x_current: numpy.ndarray with current function solutions.
    :param x_previous: numpy.ndarray with previous function solutions.
    :param dimension: Number of unique variables in the function.
    :param accuracy: Given accuracy, with which we need to find the points of minimum of the function.

    :return: Boolean value whether the current solution fulfills all the criteria.
    """

    return all([check_first_stopping_criteria(function=function,
                                              free_symbols=free_symbols,
                                              x_current=x_current,
                                              x_previous=x_previous,
                                              dimension=dimension,
                                              accuracy=accuracy),

                check_second_stopping_criteria(x_current=x_current,
                                               x_previous=x_previous,
                                               accuracy=accuracy),

                check_third_stopping_criteria(function=function,
                                              free_symbols=free_symbols,
                                              x_current=x_current,
                                              dimension=dimension,
                                              accuracy=accuracy)])
