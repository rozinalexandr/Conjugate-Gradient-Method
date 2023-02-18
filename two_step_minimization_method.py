from datetime import datetime

import numpy as np
from sympy import sympify, symbols, simplify, lambdify
from scipy.optimize import minimize_scalar

from plotter import Plotter
from stopping_criteria import StoppingCriteria
from utils import get_anti_derivative, get_derivative, get_symbol_value_mapping, get_norm_of_vector


class TwoStepMinimization:
    def __init__(self, input_list):
        self._function = sympify(input_list[0])
        self._x_0 = input_list[1]
        self._accuracy = input_list[2]
        self._min_point = input_list[3]
        self._dimension = len(self._x_0)
        self._free_symbols = sorted(self._function.free_symbols, key=lambda sym: sym.name)
        self._stopping_criteria = StoppingCriteria(self._dimension, self._function, self._free_symbols, self._accuracy)

    def _print_result_info(self, method_name, x_current, number_of_iterations, time_of_execution):
        info_str = f"Minimization Execution Results" \
                   f"\nMinimization method name: {method_name}" \
                   f"\nFunction reaches its minimum at the point: {x_current} " \
                   f"\nNumber of iterations: {number_of_iterations + 1} " \
                   f"\nTime of execution: {time_of_execution} " \
                   f"\nFunction value at the reached minimum point: {self._get_function_value_at_k_point(x_current)}" \
                   f" \nFunction value at starting point: {self._get_function_value_at_k_point(self._x_0)}\n\n"
        print(info_str)

    def _get_function_value_at_k_point(self, x_current):
        symbol_value_mapping = get_symbol_value_mapping(self._free_symbols, x_current, self._dimension)
        return self._function.subs(symbol_value_mapping)

    def _get_s_0(self):
        symbol_value_mapping = get_symbol_value_mapping(self._free_symbols, self._x_0, self._dimension)

        s_0 = []
        for i in range(self._dimension):
            s_0.append(get_anti_derivative(self._function, self._free_symbols[i]).subs(symbol_value_mapping))
        return s_0

    def _get_s_k(self, x_current, beta_current, s_previous):
        symbol_value_mapping = get_symbol_value_mapping(self._free_symbols, x_current, self._dimension)

        x_current_anti_gradient = []
        for i in range(self._dimension):
            x_current_anti_gradient.append(get_anti_derivative(self._function, self._free_symbols[i]).
                                           subs(symbol_value_mapping))
        x_current_anti_gradient_np = np.array(x_current_anti_gradient)
        s_previous_np = np.array(s_previous)

        return x_current_anti_gradient_np + beta_current * s_previous_np

    def _get_s_k_first_modification(self, x_current, beta_current, s_previous, h_previous):
        symbol_value_mapping = get_symbol_value_mapping(self._free_symbols, x_current, self._dimension)

        x_current_anti_gradient = []
        for i in range(self._dimension):
            x_current_anti_gradient.append(get_anti_derivative(self._function, self._free_symbols[i]).
                                           subs(symbol_value_mapping))
        x_current_anti_gradient_np = np.array(x_current_anti_gradient)
        s_previous_np = np.array(s_previous)
        return x_current_anti_gradient_np + beta_current * np.dot(h_previous, s_previous_np)

    def _get_s_k_second_modification(self, x_current, beta_current, s_previous, h_current):
        symbol_value_mapping = get_symbol_value_mapping(self._free_symbols, x_current, self._dimension)

        x_current_anti_gradient = []
        for i in range(self._dimension):
            x_current_anti_gradient.append(get_anti_derivative(self._function, self._free_symbols[i]).
                                           subs(symbol_value_mapping))
        x_current_anti_gradient_np = np.array(x_current_anti_gradient)
        s_previous_np = np.array(s_previous)
        return np.dot(h_current, x_current_anti_gradient_np) + np.dot(beta_current, s_previous_np)

    def _get_alpha_k(self, x_k, s_k):
        alpha = symbols("alpha")

        alpha_vector = {}
        for i in range(self._dimension):
            alpha_vector[self._free_symbols[i]] = x_k[i] + alpha * s_k[i]

        equation_with_alpa_substitution = simplify(self._function.subs(alpha_vector))

        lam_f = lambdify(alpha, equation_with_alpa_substitution)
        alpha_k = minimize_scalar(lam_f)
        return alpha_k.x

    def _get_beta_k(self, x_current, x_previous, iteration_number):
        if iteration_number % self._dimension == 0:
            return 0
        else:
            x_current_symbol_value_mapping = get_symbol_value_mapping(self._free_symbols, x_current, self._dimension)
            x_previous_symbol_value_mapping = get_symbol_value_mapping(self._free_symbols, x_previous, self._dimension)

            current_function_derivative_with_substitution = []
            for i in range(self._dimension):
                current_function_derivative_with_substitution.append(get_derivative(self._function,
                                                                                    self._free_symbols[i]).
                                                                     subs(x_current_symbol_value_mapping))

            previous_function_derivative_with_substitution = []
            for i in range(self._dimension):
                previous_function_derivative_with_substitution.append(get_derivative(self._function,
                                                                                     self._free_symbols[i]).
                                                                      subs(x_previous_symbol_value_mapping))

            return (get_norm_of_vector(current_function_derivative_with_substitution) ** 2
                    / get_norm_of_vector(previous_function_derivative_with_substitution) ** 2)

    @staticmethod
    def _get_x_next(x_current, alpha_current, s_current):
        x_np_current = np.array(x_current)
        s_np_current = np.array(s_current)
        return x_np_current + s_np_current * alpha_current

    @staticmethod
    def _get_delta_x(x_next, x_current):
        return np.subtract(x_next, x_current)

    def _get_delta_y(self, x_next, x_current):
        next_symbol_value_mapping = get_symbol_value_mapping(self._free_symbols, x_next, self._dimension)
        current_symbol_value_mapping = get_symbol_value_mapping(self._free_symbols, x_current, self._dimension)

        derivative_list = []
        for i in range(self._dimension):
            derivative_list.append(get_derivative(self._function, self._free_symbols[i]))

        next_derivative_w_substitution = []
        current_derivative_w_substitution = []
        for derivative in derivative_list:
            next_derivative_w_substitution.append(derivative.subs(next_symbol_value_mapping))
            current_derivative_w_substitution.append(derivative.subs(current_symbol_value_mapping))

        return (np.array(next_derivative_w_substitution).astype(np.float64)
                - np.array(current_derivative_w_substitution).astype(np.float64))

    def _get_h_k(self, x_next, x_current, h_previous):
        delta_x = self._get_delta_x(x_next, x_current)
        delta_y = self._get_delta_y(x_next, x_current)

        bracketed_expression = np.subtract(delta_x, np.dot(h_previous, delta_y))

        expression_right_side = np.divide(np.dot(bracketed_expression.T, bracketed_expression.T),
                                          np.dot(bracketed_expression, delta_y))

        return np.add(h_previous, expression_right_side)

    def run_base_minimizer(self):
        start_time = datetime.now()
        list_of_function_value_at_k_point = [self._get_function_value_at_k_point(self._x_0)]
        iteration_counter = 0

        s_0 = self._get_s_0()
        alpha_0 = self._get_alpha_k(self._x_0, s_0)
        x_1 = self._get_x_next(self._x_0, alpha_0, s_0)
        list_of_function_value_at_k_point.append(self._get_function_value_at_k_point(x_1))

        if (self._stopping_criteria.first_stopping_criteria(x_1, self._x_0) and
                self._stopping_criteria.second_stopping_criteria(x_1, self._x_0) and
                self._stopping_criteria.third_stopping_criteria(x_1)):
            self._print_result_info("Base Minimization", x_1, iteration_counter, datetime.now() - start_time)
        else:
            x_previous = self._x_0
            x_current = x_1
            s_previous = s_0

            while True:
                iteration_counter += 1
                beta_current = self._get_beta_k(x_current, x_previous, iteration_counter)
                s_current = self._get_s_k(x_current, beta_current, s_previous)
                alpha_current = self._get_alpha_k(x_current, s_current)
                x_next = self._get_x_next(x_current, alpha_current, s_current)
                list_of_function_value_at_k_point.append(self._get_function_value_at_k_point(x_next))

                if (self._stopping_criteria.first_stopping_criteria(x_next, x_current) and
                        self._stopping_criteria.second_stopping_criteria(x_next, x_current) and
                        self._stopping_criteria.third_stopping_criteria(x_next)):
                    self._print_result_info("Base Minimization", x_next, iteration_counter, datetime.now() - start_time)
                    break
                else:
                    x_previous = x_current
                    x_current = x_next
                    s_previous = s_current

        return list_of_function_value_at_k_point

    def run_first_modification_minimizer(self):
        list_of_function_value_at_k_point = [self._get_function_value_at_k_point(self._x_0)]
        iteration_counter = 0
        start_time = datetime.now()

        s_0 = self._get_s_0()
        alpha_0 = self._get_alpha_k(self._x_0, s_0)
        x_1 = self._get_x_next(self._x_0, alpha_0, s_0)
        h_0 = np.eye(self._dimension)
        list_of_function_value_at_k_point.append(self._get_function_value_at_k_point(x_1))

        if (self._stopping_criteria.first_stopping_criteria(x_1, self._x_0) and
                self._stopping_criteria.second_stopping_criteria(x_1, self._x_0) and
                self._stopping_criteria.third_stopping_criteria(x_1)):
            self._print_result_info("First Modification", x_1, iteration_counter, datetime.now() - start_time)
        else:
            iteration_counter += 1
            beta_1 = self._get_beta_k(x_1, self._x_0, iteration_counter)
            s_1 = self._get_s_k_first_modification(x_1, beta_1, s_0, h_0)
            alpha_1 = self._get_alpha_k(x_1, s_1)
            x_2 = self._get_x_next(x_1, alpha_1, s_1)
            h_1 = self._get_h_k(x_1, self._x_0, h_0)
            list_of_function_value_at_k_point.append(self._get_function_value_at_k_point(x_2))

            if (self._stopping_criteria.first_stopping_criteria(x_2, x_1) and
                    self._stopping_criteria.second_stopping_criteria(x_2, x_1) and
                    self._stopping_criteria.third_stopping_criteria(x_2)):
                self._print_result_info("First Modification", x_2, iteration_counter, datetime.now() - start_time)
            else:
                x_previous = x_1
                x_current = x_2
                s_previous = s_1
                h_previous = h_1

                while True:
                    iteration_counter += 1
                    beta_current = self._get_beta_k(x_current, x_previous, iteration_counter)
                    s_current = self._get_s_k_first_modification(x_current, beta_current, s_previous, h_previous)

                    alpha_current = self._get_alpha_k(x_current, s_current)
                    x_next = self._get_x_next(x_current, alpha_current, s_current)
                    list_of_function_value_at_k_point.append(self._get_function_value_at_k_point(x_next))

                    if (self._stopping_criteria.first_stopping_criteria(x_next, x_current) and
                            self._stopping_criteria.second_stopping_criteria(x_next, x_current) and
                            self._stopping_criteria.third_stopping_criteria(x_next)):
                        self._print_result_info("First Modification", x_next, iteration_counter,
                                                datetime.now() - start_time)
                        break
                    else:
                        x_previous_previous = x_previous
                        x_previous = x_current
                        x_current = x_next
                        s_previous = s_current
                        h_previous = self._get_h_k(x_previous, x_previous_previous, h_previous)
        return list_of_function_value_at_k_point

    def run_second_modification_minimizer(self):
        start_time = datetime.now()
        list_of_function_value_at_k_point = [self._get_function_value_at_k_point(self._x_0)]

        iteration_counter = 0
        s_0 = self._get_s_0()
        alpha_0 = self._get_alpha_k(self._x_0, s_0)
        x_1 = self._get_x_next(self._x_0, alpha_0, s_0)
        list_of_function_value_at_k_point.append(self._get_function_value_at_k_point(x_1))
        h_0 = np.eye(self._dimension)

        if (self._stopping_criteria.first_stopping_criteria(x_1, self._x_0) and
                self._stopping_criteria.second_stopping_criteria(x_1, self._x_0) and
                self._stopping_criteria.third_stopping_criteria(x_1)):
            self._print_result_info("Second Modification", x_1, iteration_counter, datetime.now() - start_time)
        else:
            x_previous = self._x_0
            x_current = x_1
            s_previous = s_0
            h_current = self._get_h_k(x_current, x_previous, h_0)

            while True:
                iteration_counter += 1
                beta_current = self._get_beta_k(x_current, x_previous, iteration_counter)
                s_current = self._get_s_k_second_modification(x_current, beta_current, s_previous, h_current)
                alpha_current = self._get_alpha_k(x_current, s_current)
                x_next = self._get_x_next(x_current, alpha_current, s_current)
                list_of_function_value_at_k_point.append(self._get_function_value_at_k_point(x_next))

                if (self._stopping_criteria.first_stopping_criteria(x_next, x_current) and
                        self._stopping_criteria.second_stopping_criteria(x_next, x_current) and
                        self._stopping_criteria.third_stopping_criteria(x_next)):
                    self._print_result_info("Second Modification", x_next, iteration_counter,
                                            datetime.now() - start_time)
                    break
                else:
                    x_previous = x_current
                    x_current = x_next
                    s_previous = s_current
                    h_current = self._get_h_k(x_current, x_previous, h_current)

        return list_of_function_value_at_k_point

    def run_all_methods(self):
        plotter = Plotter(self._get_function_value_at_k_point(self._min_point))

        base_function_values_list = self.run_base_minimizer()
        first_modification_values = self.run_first_modification_minimizer()
        second_modification_values = self.run_second_modification_minimizer()

        plotter.plot(base_function_values_list, first_modification_values, second_modification_values)
