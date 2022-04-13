

import numpy as np


def get_eval_points(n_electron=2, other_electron_pos=None, N_grid=16, D_min = -3, D_max = 3):
    assert N_grid%2 != 1
    
    n_dim = n_electron * 3
    # Grids for the first electron
    grids = [np.linspace(D_min, D_max, N_grid) for _ in range(3)]
    # n_main_grid = np.prod([len(arr) for arr in grids])
    # Grids for the second (other) electron
    if n_electron > 1:
        assert other_electron_pos is not None
        for pos in other_electron_pos:
            for x in pos:
                grids.extend(np.array([x]))
    meshes = np.meshgrid(*grids)
    coord = np.stack(meshes, axis=-1).reshape(-1, n_dim)
    output_shape = [N_grid for _ in range(3)]
    return coord, output_shape

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

def draw_equal_potential(density, ax, D_min, D_max, potential_level_factor = 1.0):
    N_grid = len(density[0])
    mean = np.mean(density)
    potential_level = mean * potential_level_factor
    #print(f"mean: {mean}")
    
    diff = D_max-D_min
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    try:
        verts, faces, normals, values = measure.marching_cubes(density, potential_level, spacing=tuple((diff/N_grid for _ in range(3))))
    except Exception:
        return
    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    #fig = plt.figure(figsize=(10, 10))
    #ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    #print(verts)
    #print(verts.shape)
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    ax.set_xlim(0, diff)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, diff)  # b = 10
    ax.set_zlim(0, diff)  # c = 16

    

if __name__ == "__main__1":
    from time import time
    from charge_potentials import get_nuc_ele_potential
    from pathos.multiprocessing import ProcessingPool as Pool
    coord, output_shape = get_eval_points(n_electron=1, N_grid=8, N_aux=0,D_min = -3, D_max = 3)
    def func(x):
        ans = np.exp(abs(x[2]))+np.cos(np.linalg.norm(x[0:2])*1.5)
        return ans
    with Pool(16) as p:
        potentials = np.array(p.map(func, coord)).reshape(*output_shape).reshape(*output_shape)

    draw_equal_potential(potentials, potential_level_factor = 0.1)
    
if __name__ == "__main__":
    from time import time
    from charge_potentials import get_nuc_ele_potential, ele_ele_potential
    from pathos.multiprocessing import ProcessingPool as Pool

    n_electron = 3
    nuc_ele_potential = get_nuc_ele_potential([[0.0,-1.0,0.0],[0.0,1.0,0.0]],[2,1],n_electron)
    N_grid = 30
    coord, output_shape = get_eval_points(n_electron=n_electron, other_electron_pos = [[0,1.2,0],[1.2,0,0]], N_grid=N_grid, D_min = -3, D_max = 3)
    print(coord[0])
    print("Computing")
    t0 = time()
    """
    The following code should be replaced by the thing we are computing
    """
    with Pool(16) as p:
        potentials = np.array(p.map(lambda x: nuc_ele_potential(x)+
    ele_ele_potential(x), coord)).reshape(*output_shape)
    print(f"Time used:{time()-t0}")
    print(potentials.shape)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    draw_equal_potential(potentials, ax, -3,3, potential_level_factor = 1.05)
    plt.tight_layout()
    plt.show()
    plt.savefig("test.png")