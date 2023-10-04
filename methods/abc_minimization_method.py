from abc import ABC, abstractmethod
import re


class ABCMinimisationMethod(ABC):
    def __str__(self):
        """
        For better representation and further logging the name of the method is the name of the corresponding class.
        All inherited classes must be named in the CamelCase style. This ensures that, when printed, the class name
        will be separated by capitalization using spaces.
        E.g. the name of a class is 'MyNewClass' and when we print it, we will get 'My New Class'

        :return: String with a name of a class, separated by spaces.
        """
        return " ".join(re.findall('[A-Z][^A-Z]*', type(self).__name__))

    @abstractmethod
    def run_method(self):
        pass
