from logistic_regression.functions import load_dataset, model
import matplotlib.pyplot as plt
import numpy as np

def run():
    # load the dataset
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    # flatten the arrays
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
    # standardize
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    # build the model with dfferent learning rates
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}

    for lr in learning_rates:
        print ("Training a model with learning rate: " + str(lr))
        models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=10000, learning_rate=lr, print_cost=False)
        print ('\n' + "-------------------------------------------------------" + '\n')

    # plot
    for lr in learning_rates:
        plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()