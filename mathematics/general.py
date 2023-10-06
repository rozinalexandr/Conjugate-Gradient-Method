import sympy
import numpy as np

from typing import List, Dict


def get_derivative(function: sympy.core.add.Add,
                   symbol: sympy.Symbol) -> sympy.core.add.Add:
    """
    This function takes the derivative of origin function which was transformed with sympy.sympify()

    :param function: Function, which was transformed with sympy.sympify().
    :param symbol: sympy.Symbol - unique origin function variable.

    :return: Function derivative.
    """

    return sympy.diff(function, symbol)


def get_anti_derivative(function: sympy.core.add.Add,
                        symbol: sympy.Symbol) -> sympy.core.add.Add:
    """
    This function takes the antiderivative of origin function which was transformed with sympy.sympify()

    :param function: Function, which was transformed with sympy.sympify().
    :param symbol: sympy.Symbol - unique origin function variable.

    :return: Function antiderivative.
    """

    return -sympy.diff(function, symbol)


def get_norm_of_vector(vector: np.ndarray[float | int]) -> float:
    """
    Function calculates the norm of given vector.

    :param vector: np.ndarray with float or int values.

    :return: Norm of given vector.
    """
    return sum([abs(element) ** 2 for element in vector]) ** 0.5


def get_symbol_value_mapping(symbols_array: List[sympy.Symbol],
                             values_array: np.ndarray[float | int],
                             dimension: int) -> Dict[sympy.Symbol, float | int]:
    """
    Function maps the variables of origin function to the values of those variables. The output is a dictionary in
    which the key is sympy.Symbol and the value is the corresponding value to this variable.
    E.g. origin function has 2 variables: x_1, x_2. After some calculations, we found out that their values are [0, 1]
    respectively. SO, the output dictionary will be as follows: {"x_1": 0, "x_2": 1}. This mapping is needed in further
    calculations.

    :param symbols_array: List, which contains unique sympy.Symbols of variables from origin function.
    :param values_array: np.ndarray with values of origin function variables.
    :param dimension: Number of unique function variables.

    :return: Dictionary with {symbol: value} mapping.
    """

    symbol_value_mapping = {}
    for i in range(dimension):
        symbol_value_mapping[symbols_array[i]] = values_array[i]

    return symbol_value_mapping


def get_function_value_at_k_point(function: sympy.core.add.Add,
                                  free_symbols: List[sympy.Symbol],
                                  x_current: np.ndarray[float | int],
                                  dimension: int) -> int | float:
    """
    Function calculates value of the origin function with current coordinates.

    :param function: Function, which was transformed with sympy.sympify().
    :param free_symbols: List, which contains unique sympy.Symbols of variables from origin function.
    :param x_current: np.ndarray with values of origin function variables.
    :param dimension: Number of unique function variables.

    :return: Function value at current coordinates.
    """

    symbol_value_mapping = get_symbol_value_mapping(free_symbols, x_current, dimension)
    return function.subs(symbol_value_mapping)
