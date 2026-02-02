from pathlib import Path

from lib.data import load_iris
from lib.gmm import train_gmm
from solution.gmm import save_gmm

DATA_PATH = Path(__file__).parent / "data"


def main():
    # train_data_path = DATA_PATH / "GMM_data_4D.npy"
    # train_data = load(train_data_path)
    # gmm_path = DATA_PATH / "GMM_4D_3G_init.json"
    # gmm_path = DATA_PATH / "GMM_4D_4G_EM_LBG.json"
    # gmm = load_gmm(gmm_path)

    train_data, train_target, test_data, test_target = load_iris()

    min_eig = 0.01
    stop_delta = 1e-6
    alpha = 0.1
    target_components = 4
    my_gmm = train_gmm(train_data, alpha, stop_delta, target_components)
    save_gmm(my_gmm, DATA_PATH / "my_gmm.json")


if __name__ == "__main__":
    main()
