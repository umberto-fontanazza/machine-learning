def load_data() -> tuple[list[str], list[str], list[str]]:

    lInf = []

    f = open("data/inferno.txt", encoding="ISO-8859-1")

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    f = open("data/purgatorio.txt", encoding="ISO-8859-1")

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    f = open("data/paradiso.txt", encoding="ISO-8859-1")

    for line in f:
        lPar.append(line.strip())
    f.close()

    return lInf, lPur, lPar


def split_data(lines: list[str], fraction: int = 4) -> tuple[list[str], list[str]]:

    lTrain, lTest = [], []
    for i in range(len(lines)):
        if i % fraction == 0:
            lTest.append(lines[i])
        else:
            lTrain.append(lines[i])

    return lTrain, lTest


if __name__ == "__main__":

    # Load the tercets and split the lists in training and test lists

    lInf, lPur, lPar = load_data()

    lInf_train, lInf_evaluation = split_data(lInf, 4)
    lPur_train, lPur_evaluation = split_data(lPur, 4)
    lPar_train, lPar_evaluation = split_data(lPar, 4)

    print(lInf_train)
