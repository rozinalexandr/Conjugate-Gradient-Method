from methods.conjugate_gradients import ConjugateGradients
from methods.conjugate_gradients_1st_modification import ConjugateGradientsFirstModification
from methods.conjugate_gradients_2nd_modification import ConjugateGradientsSecondModification
from methods.conjugate_gradients_3rd_modification import ConjugateGradientsThirdModification
from drawing.plotter import make_plot


class Controller:
    def __init__(self, settings: dict):
        self._settings = settings

    @staticmethod
    def _get_minimization_method(method_name: str):
        """
        Using the Factory Method design pattern, we simplify the process of creating instances of minimization method
        classes. Thus, only those instances of minimization methods that are needed will be created.
        In order to add a new method, it is necessary to specify the method name as the key and the method class as
        the value in the methods_dict variable.

        :param method_name: String parameter with the name of the method.
        :return: Instance of required method class.
        """
        methods_dict = {
            "Conjugate Gradients": ConjugateGradients,
            "Conjugate Gradients 1st Modification": ConjugateGradientsFirstModification,
            "Conjugate Gradients 2nd Modification": ConjugateGradientsSecondModification,
            "Conjugate Gradients 3rd Modification": ConjugateGradientsThirdModification
        }

        try:
            return methods_dict[method_name]()
        except KeyError:
            raise KeyError(f"Wrong Method Name: {method_name}")

    def run(self):
        # selecting all the methods, which have 'True' value
        selected_methods = [k for k, v in self._settings["Methods Selection"].items() if v]

        for selected_method_name in selected_methods:
            method = self._get_minimization_method(method_name=selected_method_name)
            method.run_method()



        test1 = {"name": "Conjugate Gradients", "x": [0, 1, 2, 3, 4], "y": [0, 1, 2, 3, 4]}
        test2 = {"name": "Conjugate Gradients 1st Modification", "x": [0, 1, 2, 3, 4], "y": [0, 10, 20, 30, 40]}
        test3 = {"name": "Conjugate Gradients 2nd Modification", "x": [0, 10, 20, 30, 40], "y": [0, 1, 2, 3, 4]}
        test4 = {"name": "Conjugate Gradients 3rd Modification", "x": [40, 31, 22, 13, 4], "y": [0, 10, 20, 30, 40]}
        test5 = {"name": "Test", "x": [10, 20, 30, 40, 50], "y": [20, 20, 20, 20, 20]}
        if self._settings["Plotter Settings"]["Plot"]:
            make_plot(test1, test2, test3, test4, test5)
