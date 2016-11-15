from util import readFile
import plotly.plotly as py
from plotly.graph_objs import *

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
    loss = data[1]

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
        title = "Training Total Loss vs. Epoch (%s activation)" % activation,
        xaxis = dict(title = "Epoch"),
        yaxis = dict(title = "Training Total Loss")
    )

    fig = dict(
        data = data,
        layout = layout
    )
    py.plot(fig, filename="model selection")

def plot_classify_model(dataset, activation):
    lam = 0.4
    iterations = 500
    data = []

    epochs = range(1,iterations+1)
    for alpha in [0.01, 0.05]:
        for layer_dim in [100, 200, 400]:
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
    plot_classify("classify", "sigmoid", 0.01, 0.4, 200, 500)
    #plot_classify_model("classify", "sigmoid")


if __name__ == '__main__':
    main()