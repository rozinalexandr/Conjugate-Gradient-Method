from controller import Controller


settings = {
    "Function Settings": {
        "Function": "(x_1**2 + x_2 - 11)**2 + (x_1 + x_2**2 - 7)**2",
        "Starting Coordinates": [1, 1],
        "Accuracy": -5,
        "Specified Minimum Coordinates": [3, 2],
        "Iteration Threshold": 1000000
    },

    "Methods Selection": {
        "Conjugate Gradients": True,
        "Conjugate Gradients 1st Modification": True,
        "Conjugate Gradients 2nd Modification": True,
        "Conjugate Gradients 3rd Modification": False,
        "Conjugate Gradients 4th Modification": False
    },

    "Alpha k Selection": {
        "Single-Factor Minimization": True,
        "Doubling Method": False
    },

    "Plotter Settings": {
        "Plot": True
    }
}

input_lst = [
    "100*(x_2-x_1**2)**2 + (1-x_1)**2 + 100*(x_4-x_3**2)**2 + (1-x_3)**2 + 100*(x_6-x_5**2)**2 + (1-x_5)**2 + 100*(x_8-x_7**2)**2 + (1-x_7)**2", [-1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1], -7, [1,1,1,1, 1,1,1,1]
]
new_settings = settings["Function Settings"]
new_settings['Function'] = input_lst[0]
new_settings['Starting Coordinates'] = input_lst[1]
new_settings['Accuracy'] = input_lst[2]
new_settings['Specified Minimum Coordinates'] = input_lst[3]
new_settings['Iteration Threshold'] = 1000000
print(settings)

controller = Controller(settings)
controller.run()
