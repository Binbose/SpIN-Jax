from train_spin import ModelTrainer
from helper import get_exact_eigenvalues
import numpy as np
from multiprocessing import Pool, Process
import sys
from pathlib import Path
import json

def save_result(key , value, save_dir="./"):
    save_file_dir = save_dir+"/param_phase_diagram.json"
    existing_dict = {}
    if Path(save_file_dir).exists():
        with open(save_file_dir,"r") as f:
            existing_dict = json.load(f)
    existing_dict[key] = value
    with open(save_file_dir,"w") as f:
        json.dump(existing_dict, f)

def get_N_epoches_to_converge(moving_average_beta = 1, learning_rate = 1e-5, error_threshold = 0.5):

    trainer = ModelTrainer()
    trainer.num_epochs = 2000
    trainer.moving_average_beta = moving_average_beta
    trainer.learning_rate = learning_rate

    ground_truth = get_exact_eigenvalues(trainer.system,trainer.n_eigenfuncs,trainer.n_space_dimension,trainer.D_min, trainer.D_max, trainer.charge)

    result_dict = {"n_epoch_to_threshold": -1}

    def callback(epoch, **kwargs):
        energies = np.asarray(kwargs["energies"])
        newest_energy = np.average(energies[-trainer.window:],axis=0)
        energy_error = abs(np.average(newest_energy - ground_truth))
        print(moving_average_beta, learning_rate, energy_error)
        if energy_error <= error_threshold:
            result_dict["n_epoch_to_threshold"] = epoch
            save_result((moving_average_beta,learning_rate,error_threshold),epoch)
            return True

    trainer.start_training(show_progress=False, callback = callback)

    print(result_dict)

if __name__ == "__main__":
    n_points = 5
    moving_average_beta = np.ones(n_points)-np.ones(n_points)/np.linspace(20,100,n_points)
    moving_average_beta = [1]
    learning_rate = 1e-5+5e-6*np.linspace(-1,1,n_points)
    param_list = []
    error_threshold = 1.5
    for beta in moving_average_beta:
        for rate in learning_rate:
            param_list.append((beta,rate,error_threshold))
    for param in param_list:
        get_N_epoches_to_converge(*param)
