from util import readFile
import plotly.plotly as py
from plotly.graph_objs import *

parameters = dict(
    classify = dict(
        iterations = 500,
        sigmoid = dict(
            alpha = [0.01, 0.05],
            dim = [100, 200, 400]
        ),
        relu = dict(
            alpha = [0.01, 0.05],
            dim = [100, 200, 400]
        )
    ),
    regression = dict(
        iterations = 300,
        sigmoid = dict(
            alpha = [0.01, 0.001],
            dim = [3, 4, 5]
        ),
        relu = dict(
            alpha = [0.001, 0.005],
            dim = [3, 4, 5]
        )
    )
)

def load_data(file_path):
    lines = readFile(file_path).strip().split("\n")
    results = [tuple([float(n) for n in line.strip().split()]) for line in lines]
    results = zip(*results)
    return map(list, results)



def plot_classify(dataset, activation, alpha, lam, layer_dim, iterations):
    file_name = "_".join([dataset,
                          activation,
                          str(alpha),
                          str(lam),
                          str(layer_dim),
                          str(iterations)]) + ".txt"
    file_path = "result/" + file_name
    data = load_data(file_path)
    epochs = data[0]
    loss = data[2]

    trace0 = Scatter(
        x = epochs,
        y = loss,
        line = dict(
            width = 3,
        ),
        mode = 'lines',
        name = " ".join([dataset, activation, str(alpha), str(layer_dim)]),
    )
    data = [trace0]

    # Plot and embed in ipython notebook!
    layout = dict(
        title = "Validation Total Loss vs. Epoch (%s activation)" % activation,
        xaxis = dict(title = "Epoch"),
        yaxis = dict(title = "Validation Total Loss")
    )

    fig = dict(
        data = data,
        layout = layout
    )
    py.plot(fig, filename="model selection")

def plot_classify_model(dataset, activation):
    lam = 0.4
    iterations = parameters[dataset]["iterations"]
    data = []

    epochs = range(1,iterations+1)
    for alpha in parameters[dataset][activation]["alpha"]:
        for layer_dim in parameters[dataset][activation]["dim"]:
            file_name = "_".join([dataset,
                          activation,
                          str(alpha),
                          str(lam),
                          str(layer_dim),
                          str(iterations)]) + ".txt"
            file_path = "result/" + file_name
            d = load_data(file_path)
            val_loss = d[2]
            trace = Scatter(
                x = epochs,
                y = val_loss,
                mode = "lines",
                name = " ".join([dataset, activation, str(alpha), str(layer_dim)]),
                line = dict(
                    width = 2,
                )
            )
            data.append(trace)

    layout = dict(
        title = "Validation Total Loss vs. Epoch (%s activation)" % activation,
        xaxis = dict(title = "Epoch"),
        yaxis = dict(title = "Validation Total Loss")
    )

    fig = dict(
        data = data,
        layout = layout
    )
    py.plot(fig, filename="model selection")


def main():
    #plot_classify("regression", "sigmoid", 0.001, 0.4, 3, 300)
    plot_classify_model("regression", "relu")


if __name__ == '__main__':
    main()