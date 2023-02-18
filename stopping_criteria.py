from sympy import sympify
from utils import get_norm_of_vector, get_derivative, get_symbol_value_mapping


class StoppingCriteria:
    def __init__(self, dimension, function, free_symbols, accuracy):
        self._function = sympify(function)
        self._accuracy = accuracy
        self._dimension = dimension
        self._free_symbols = free_symbols

    def first_stopping_criteria(self, x_current, x_previous):
        previous_symbol_value_mapping = get_symbol_value_mapping(self._free_symbols, x_previous, self._dimension)
        current_symbol_value_mapping = get_symbol_value_mapping(self._free_symbols, x_current, self._dimension)

        previous_function_solution = self._function.subs(previous_symbol_value_mapping)
        current_function_solution = self._function.subs(current_symbol_value_mapping)
        return (previous_function_solution - current_function_solution
                < self._accuracy * (1 + abs(current_function_solution)))

    def second_stopping_criteria(self, x_current, x_previous):
        return get_norm_of_vector(x_previous - x_current) < self._accuracy ** 0.5 * (1 + get_norm_of_vector(x_current))

    def third_stopping_criteria(self, x_current):
        symbol_value_mapping = get_symbol_value_mapping(self._free_symbols, x_current, self._dimension)

        function_derivative_with_substitution = []
        for i in range(self._dimension):
            function_derivative_with_substitution.append(get_derivative(self._function, self._free_symbols[i]).
                                                         subs(symbol_value_mapping))

        current_function_solution = self._function.subs(symbol_value_mapping)

        return (get_norm_of_vector(function_derivative_with_substitution)
                <= self._accuracy ** (1 / 3) * (1 + abs(current_function_solution)))
