
# SpIN Jax

This repository provides a jax implementation of 
[Spectral Inference Networks: Unifying Deep and Spectral Learning](https://arxiv.org/abs/1806.02215). 
It is not the authors official tensorflow implementation which you can find [here](https://github.com/deepmind/spectral_inference_networks), 
but the results are very similar, and run thanks to jax's jit even a bit faster. However, at the moment it only supports Hamiltonian systems.
If you are interested in general kernels you will need to add some small extra modifications.  

Additionally, we provide a deep dive with a breakdown of the code, some extra explanations and additional experiments visualizations in this [colab notebook](https://colab.research.google.com/drive/1hRm3zbf8ptJ00dGKKTohtBL3WNIg7tEl?usp=sharing#scrollTo=0PiLKO_bQjvp).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hRm3zbf8ptJ00dGKKTohtBL3WNIg7tEl?usp=sharing#scrollTo=0PiLKO_bQjvp)

## Installation
You will need jax, tensorflow (for checkpointing) and flax (for the neural networks) along with a few other requirements. To get started run
```
conda create --name spin --file requirements.txt
conda activate spin
```

## Usage
To run the training of SpIN for the wave functions of the hydrogen atoms just run
```
python train_spin.py
```
The hyperparameter are found and can be modified within ```the train_spin.py``` file.
For now only ```system='laplacian'``` and ```system='hydrogen'``` are supported. 
If you want to give it a try for different Hamiltonian system you can add the corresponding potentials in ```physics.py```, and increase ```n_space_dimension``` for multi particle systems accordingly.

## Results
The following shows the training evolution for the first 4 eigenfunctions of the hydrogen atom. 

[![Sample training](https://img.youtube.com/vi/6RKY7s5z_b4/0.jpg)](https://www.youtube.com/watch?v=6RKY7s5z_b4)

