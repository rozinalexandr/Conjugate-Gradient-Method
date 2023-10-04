from controller import Controller


settings = {
    "Methods Selection": {
        "Conjugate Gradients": True,
        "Conjugate Gradients 1st Modification": True,
        "Conjugate Gradients 2nd Modification": True,
        "Conjugate Gradients 3rd Modification": True
    },

    "Alpha k Selection": {
        "Single-Factor Minimization": True,
        "Doubling Method": False
    }
}


controller = Controller(settings)
controller.run()
