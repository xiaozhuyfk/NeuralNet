import numpy as np

# Read a file
# filename is the path of the file, string type
# returns the content as a string
def readFile(filename, mode = "rt"):
    # rt stands for "read text"
    fin = contents = None
    try:
        fin = open(filename, mode)
        contents = fin.read()
    finally:
        if (fin != None): fin.close()
    return contents


# Write 'contents' to the file
# 'filename' is the path of the file, string type
# 'contents' is of string type
# returns True if the content has been written successfully
def writeFile(filename, contents, mode = "wt"):
    # wt stands for "write text"
    fout = None
    try:
        fout = open(filename, mode)
        fout.write(contents)
    finally:
        if (fout != None): fout.close()
    return True

def load_mnist_X(path):
    lines = readFile(path).strip().split("\n")
    X = [[int(n) for n in line.strip().split(",")] for line in lines]
    return np.array(X, dtype=np.float32)

def load_mnist_Y(path):
    lines = readFile(path).strip().split("\n")
    Y = [[(int(n) == 0) * 1, (int(n) == 1) * 1] for n in lines]
    return np.array(Y, dtype=np.float32)

def load_regression_X(path):
    lines = readFile(path).strip().split("\n")
    X = [[float(n) for n in line.strip().split()] for line in lines]
    return np.array(X, dtype=np.float32)

def load_regression_Y(path):
    lines = readFile(path).strip().split("\n")
    Y = [float(n) for n in lines]
    return np.array(Y, dtype=np.float32)

def z_norm(X):
    m, n = X.shape
    means = X.sum(axis = 0) / float(m)
    variance = ((X - means)**2).sum(axis = 0) / float(m)
    stdev = variance**0.5
    for i in xrange(n):
        if stdev[i] == 0:
            stdev[i] = 1
    return (X - means) / stdev
