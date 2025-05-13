from csv import reader


def load():
    with open("./iris.csv", "r") as file:
        csv_file = reader(file)
        for row in csv_file:
            features, label = row[0:4], row[4]
            print(f"{features=}\n{label=}")


def main():
    load()


if __name__ == "__main__":
    main()
