from methods.abc_minimization_method import ABCMinimisationMethod
from typing import List, Callable


class ConjugateGradientsSecondModification(ABCMinimisationMethod):
    def __init__(self,
                 function: str,
                 x_0: List[int | float],
                 min_point: List[int | float],
                 accuracy: int,
                 threshold: int,
                 alpha_k_calculating_method: Callable):
        """
        All the minimization methods shares the same input data.

        :param function: Function in a string format.
        :param x_0: List with starting coordinates.
        :param min_point: List with known minimum coordinates.
        :param accuracy: Given accuracy, with which we need to find the points of minimum of the function.
        :param threshold: Number of iterations after which the minimization process terminates.
        :param alpha_k_calculating_method: Function reference for alpha calculation.
        :param dimension: Number of unique variables in the function.
        :param free_symbols: List of unique sympy.Symbols, which are the function variables.
        """
        super().__init__(function, x_0, min_point, accuracy, threshold, alpha_k_calculating_method)

    def run_method(self):
        print(f"{self} initialized with following parameters: {vars(self)}")
