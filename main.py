from two_step_minimization_method import TwoStepMinimization

# Examples of three minimization methods
input_lst = ["(x_1**2 + x_2 - 11)**2 + (x_1 + x_2**2 - 7)**2", [1, 1], 0.000001, [3, 2]]
# input_lst = ["(x_2 - x_1**2)**2 + 100*(1-x_1)**2", [-1.2, 1], 0.000001, [1, 1]]
# input_lst = ["-x_1**2 * exp(1-x_1**2-20.25*(x_1-x_2)**2)", [0.1, 0.1], 0.000001, [-1, -1]]
# input_lst = ["(x_2 - x_1**2)**2 + (1-x_1)**2", [-1.2, 1], 0.000001, [1, 1]]
# input_lst = ["(1.5-x_1*(1-x_2))**2+(2.25-x_1*(1-x_2**2))**2+(2.625-x_1*(1-x_2**3))**2", [2, 0.2], 0.000001, [3, 0.5]]

# Example of too long runtime of the 2nd modification
# input_lst = ["100*(x_2 - x_1**2) ** 2 + (1 - x_1)**2", [-1.2, 1], 0.000001, [1, 1]]

# example of entrapment
# input_lst = ["(x_1+10*x_2)**2+5*(x_3-x_4)**2+(x_2-2*x_3)**4+10*(x_1-x_4)**4", [3, -1, 0, 1], 0.0000000001, [0, 0, 0, 0]]

minimizer = TwoStepMinimization(input_lst)
minimizer.run_all_methods()
