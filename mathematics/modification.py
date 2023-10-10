import numpy as np
import sympy
from sympy import symbols, simplify, lambdify
from scipy.optimize import minimize_scalar

from typing import List

from mathematics.general import get_symbol_value_mapping, get_anti_derivative, get_derivative, get_norm_of_vector


def get_alpha_k_single_factor_minimization(function: sympy.core.add.Add,
                                           free_symbols: List[sympy.Symbol],
                                           x_current: np.ndarray[float | int],
                                           s_current: np.ndarray[float | int],
                                           dimension: int) -> float | int:
    """
    The function performs single factor minimization of the original function with respect to the variable alpha.

    :param function: Function, which was transformed with sympy.sympify().
    :param free_symbols: List, which contains unique sympy.Symbols of variables from origin function.
    :param x_current: np.ndarray with values of origin function variables at current point.
    :param s_current: np.ndarray with values of descent direction vector.
    :param dimension: Number of unique function variables.

    :return: Value of alpha variable.
    """

    alpha = symbols("alpha")

    alpha_vector = {}
    for i in range(dimension):
        alpha_vector[free_symbols[i]] = x_current[i] + alpha * s_current[i]

    equation_with_alpa_substitution = simplify(function.subs(alpha_vector))

    lam_f = lambdify(alpha, equation_with_alpa_substitution)
    alpha_k = minimize_scalar(lam_f)
    return alpha_k.x


def get_alpha_k_doubling_method():
    print("Alpha is calculated with Doubling Method")


def get_x_next(x_current: np.ndarray[float | int],
               alpha_current: float | int,
               s_current: np.ndarray[float | int]) -> np.ndarray[float | int]:
    """
    The function calculates the next point of approach to the minimum.

    :param x_current: np.ndarray with values of origin function variables at current point.
    :param alpha_current: Value of alpha variable.
    :param s_current: np.ndarray with values of descent direction vector at current point.

    :return: The next point coordinates.
    """

    return x_current + s_current * alpha_current


def get_beta_k(function: sympy.core.add.Add,
               free_symbols: List[sympy.Symbol],
               x_current: np.ndarray[float | int],
               x_previous: np.ndarray[float | int],
               iteration_number: int,
               dimension: int) -> int | float:
    """
    The function calculates the descent step size using the update procedure (see more information if README).

    :param function: Function, which was transformed with sympy.sympify().
    :param free_symbols: List, which contains unique sympy.Symbols of variables from origin function.
    :param x_current: np.ndarray with values of origin function variables at current point.
    :param x_previous: np.ndarray with values of origin function variables at previous point.
    :param iteration_number: Number of iterations.
    :param dimension: Number of unique function variables.

    :return: Value of descent step size.
    """

    if iteration_number % dimension == 0:
        return 0

    else:
        x_current_symbol_value_mapping = get_symbol_value_mapping(free_symbols, x_current, dimension)
        x_previous_symbol_value_mapping = get_symbol_value_mapping(free_symbols, x_previous, dimension)

        current_function_derivative_with_substitution = []
        previous_function_derivative_with_substitution = []
        for i in range(dimension):
            current_function_derivative_with_substitution.append(get_derivative(function, free_symbols[i]).
                                                                 subs(x_current_symbol_value_mapping))

            previous_function_derivative_with_substitution.append(get_derivative(function, free_symbols[i])
                                                                  .subs(x_previous_symbol_value_mapping))

        return (get_norm_of_vector(np.array(current_function_derivative_with_substitution)) ** 2
                / get_norm_of_vector(np.array(previous_function_derivative_with_substitution)) ** 2)


def get_s_k_base_modification(function: sympy.core.add.Add,
                              free_symbols: List[sympy.Symbol],
                              x_current: np.ndarray[float | int],
                              beta_current: int | float,
                              s_previous: np.ndarray[float | int],
                              dimension: int,
                              iteration: int) -> np.ndarray[float | int]:
    """
    The function calculates current direction of descent (see more information in README).

    :param function: Function, which was transformed with sympy.sympify().
    :param free_symbols: List, which contains unique sympy.Symbols of variables from origin function.
    :param x_current: np.ndarray with values of origin function variables at current point.
    :param beta_current: Value of descent step size.
    :param s_previous: np.ndarray with values of descent direction vector at previous point.
    :param dimension: Number of unique function variables.
    :param iteration: Number of current iteration.

    :return: np.ndarray with values of descent direction vector.
    """

    symbol_value_mapping = get_symbol_value_mapping(free_symbols, x_current, dimension)

    if iteration == 0:
        s_0 = []
        for i in range(dimension):
            s_0.append(get_anti_derivative(function, free_symbols[i]).subs(symbol_value_mapping))
        return np.array(s_0)

    else:
        x_current_anti_gradient = []
        for i in range(dimension):
            x_current_anti_gradient.append(get_anti_derivative(function, free_symbols[i]).subs(symbol_value_mapping))

        x_current_anti_gradient_np = np.array(x_current_anti_gradient)

        return x_current_anti_gradient_np + beta_current * s_previous


def get_delta_x(x_next: np.ndarray[float | int],
                x_current: np.ndarray[float | int]) -> np.ndarray[float | int]:
    """
    The function calculates the difference of coordinates of points x. For more information see README.

    :param x_next: np.ndarray with values of origin function variables at next point.
    :param x_current: np.ndarray with values of origin function variables at current point.

    :return: np.ndarray with coordinate difference.
    """

    return np.subtract(x_next, x_current)


def get_delta_y(function: sympy.core.add.Add,
                free_symbols: List[sympy.Symbol],
                x_next: np.ndarray[float | int],
                x_current: np.ndarray[float | int],
                dimension: int) -> np.ndarray[float | int]:
    """
    Function calculates the special parameter delta Y. For more information see README.

    :param function: Function, which was transformed with sympy.sympify().
    :param free_symbols: List, which contains unique sympy.Symbols of variables from origin function.
    :param x_next: np.ndarray with values of origin function variables at next point.
    :param x_current: np.ndarray with values of origin function variables at current point.
    :param dimension: Number of unique function variables.

    :return: np.ndarray with values of delta Y.
    """

    next_symbol_value_mapping = get_symbol_value_mapping(free_symbols, x_next, dimension)
    current_symbol_value_mapping = get_symbol_value_mapping(free_symbols, x_current, dimension)

    derivative_list = []
    for i in range(dimension):
        derivative_list.append(get_derivative(function, free_symbols[i]))

    next_derivative_w_substitution = []
    current_derivative_w_substitution = []
    for derivative in derivative_list:
        next_derivative_w_substitution.append(derivative.subs(next_symbol_value_mapping))
        current_derivative_w_substitution.append(derivative.subs(current_symbol_value_mapping))

    return (np.array(next_derivative_w_substitution).astype(np.float64)
            - np.array(current_derivative_w_substitution).astype(np.float64))


def get_h_k(function: sympy.core.add.Add,
            free_symbols: List[sympy.Symbol],
            x_next: np.ndarray[float | int],
            x_current: np.ndarray[float | int],
            dimension: int,
            h_previous: np.ndarray[np.ndarray[float | int]]) -> np.ndarray[np.ndarray[float | int]]:
    """
    Function calculates a special matrix H, which is used in modifications. For more information see README.

    :param function: Function, which was transformed with sympy.sympify().
    :param free_symbols: List, which contains unique sympy.Symbols of variables from origin function.
    :param x_next: np.ndarray with values of origin function variables at next point.
    :param x_current: np.ndarray with values of origin function variables at current point.
    :param dimension: Number of unique function variables.
    :param h_previous: Matrix H, which was calculated at previous iteration.

    :return: Special matrix H.
    """

    delta_x = get_delta_x(x_next=x_next,
                          x_current=x_current)

    delta_y = get_delta_y(function=function,
                          free_symbols=free_symbols,
                          x_next=x_next,
                          x_current=x_current,
                          dimension=dimension)

    bracketed_expression = np.subtract(delta_x, np.dot(h_previous, delta_y))

    expression_right_side = np.divide(np.dot(bracketed_expression.T, bracketed_expression.T),
                                      np.dot(bracketed_expression, delta_y))

    return np.add(h_previous, expression_right_side)


def get_s_k_second_modification(function: sympy.core.add.Add,
                                free_symbols: List[sympy.Symbol],
                                x_current: np.ndarray[float | int],
                                beta_current: float | int,
                                s_previous: np.ndarray[float | int],
                                h_current: np.ndarray[np.ndarray[float | int]],
                                dimension: int,
                                iteration: int) -> np.ndarray[float | int]:
    """
    The function calculates current direction of descent (see more information in README).

    :param function: Function, which was transformed with sympy.sympify().
    :param free_symbols: List, which contains unique sympy.Symbols of variables from origin function.
    :param x_current: np.ndarray with values of origin function variables at current point.
    :param beta_current: Value of descent step size.
    :param s_previous: np.ndarray with values of descent direction vector at previous point.
    :param h_current: Special matrix H, used for this modification.
    :param dimension: Number of unique function variables.
    :param iteration: Number of current iteration.

    :return: np.ndarray with values of descent direction vector.
    """

    symbol_value_mapping = get_symbol_value_mapping(free_symbols, x_current, dimension)

    if iteration == 0:
        s_0 = []
        for i in range(dimension):
            s_0.append(get_anti_derivative(function, free_symbols[i]).subs(symbol_value_mapping))
        return np.array(s_0)

    else:
        x_current_anti_gradient = []
        for i in range(dimension):
            x_current_anti_gradient.append(get_anti_derivative(function, free_symbols[i]).subs(symbol_value_mapping))

        x_current_anti_gradient_np = np.array(x_current_anti_gradient)
        s_previous_np = np.array(s_previous)

        return np.dot(h_current, x_current_anti_gradient_np) + np.dot(beta_current, s_previous_np)


def get_s_k_first_modification(function: sympy.core.add.Add,
                               free_symbols: List[sympy.Symbol],
                               x_current: np.ndarray[float | int],
                               beta_current: float | int,
                               s_previous: np.ndarray[float | int],
                               h_previous: np.ndarray[np.ndarray[float | int]],
                               dimension: int,
                               iteration: int) -> np.ndarray[float | int]:
    """
    The function calculates current direction of descent (see more information in README).

    :param function: Function, which was transformed with sympy.sympify().
    :param free_symbols: List, which contains unique sympy.Symbols of variables from origin function.
    :param x_current: np.ndarray with values of origin function variables at current point.
    :param beta_current: Value of descent step size.
    :param s_previous: np.ndarray with values of descent direction vector at previous point.
    :param h_previous: Special matrix H, used for this modification.
    :param dimension: Number of unique function variables.
    :param iteration: Number of current iteration.

    :return: np.ndarray with values of descent direction vector.
    """

    symbol_value_mapping = get_symbol_value_mapping(free_symbols, x_current, dimension)

    if iteration == 0:
        s_0 = []
        for i in range(dimension):
            s_0.append(get_anti_derivative(function, free_symbols[i]).subs(symbol_value_mapping))
        return np.array(s_0)

    else:
        x_current_anti_gradient = []
        for i in range(dimension):
            x_current_anti_gradient.append(get_anti_derivative(function, free_symbols[i]).subs(symbol_value_mapping))

        x_current_anti_gradient_np = np.array(x_current_anti_gradient)
        s_previous_np = np.array(s_previous)

        return x_current_anti_gradient_np + beta_current * np.dot(h_previous, s_previous_np)
