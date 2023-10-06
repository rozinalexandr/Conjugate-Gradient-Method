from controller import Controller


settings = {
    "Function Settings": {
        "Function": "(x_1**2 + x_2 - 11)**2 + (x_1 + x_2**2 - 7)**2",
        "Starting Coordinates": [1, 1],
        "Accuracy": -5,
        "Specified Minimum Coordinates": [3, 2],
        "Iteration Threshold": 1000
    },

    "Methods Selection": {
        "Conjugate Gradients": True,
        "Conjugate Gradients 1st Modification": False,
        "Conjugate Gradients 2nd Modification": False,
        "Conjugate Gradients 3rd Modification": False
    },

    "Alpha k Selection": {
        "Single-Factor Minimization": True,
        "Doubling Method": False
    },

    "Plotter Settings": {
        "Plot": True
    }
}


controller = Controller(settings)
controller.run()
