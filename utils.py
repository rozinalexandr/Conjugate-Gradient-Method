from sympy import diff


def get_derivative(function, symbol):
    return diff(function, symbol)


def get_anti_derivative(function, symbol):
    return -diff(function, symbol)


def get_norm_of_vector(vector):
    return sum([abs(element) ** 2 for element in vector]) ** 0.5


def get_symbol_value_mapping(symbols_array, values_array, dimension):
    symbol_value_mapping = {}
    for i in range(dimension):
        symbol_value_mapping[symbols_array[i]] = values_array[i]
    return symbol_value_mapping
