import numpy as np

def ele_ele_potential(x):
    n_electron = len(x)//3
    x_= np.expand_dims(x.reshape(n_electron,3), axis=0)
    ele_xs_1 = np.tile(x_,(n_electron,1,1))
    ele_xs_2 = np.tile(np.transpose(x_,(1,0,2)),(1,n_electron,1))
    diff = ele_xs_1-ele_xs_2
    norms = np.linalg.norm(diff,axis=(2))
    norms_inv = 1/(norms+np.eye(n_electron))-np.eye(n_electron)
    v_ele_ele = np.sum(norms_inv,axis=(0,1))/2
    return v_ele_ele

def get_nuc_ele_potential(nuc_x,nuc_charge,n_electron):
    n_nuc = len(nuc_x)
    nuc_xs = np.tile(np.transpose(np.expand_dims(nuc_x, axis=0),(1,0,2)),(1,n_electron,1))
    def nuc_ele_potential(x):
        ele_xs = np.tile(np.expand_dims(x.reshape(len(x)//3,3), axis=0),(n_nuc,1,1))
        diff = nuc_xs - ele_xs
        norms = np.linalg.norm(diff,axis=(2))
        norms_inv = 1/norms
        norms_with_charge = np.multiply(norms_inv.transpose(),nuc_charge)
        v_nuc_ele = -np.sum(norms_with_charge,axis=(0,1))
        return v_nuc_ele
    return nuc_ele_potential

def nuc_nuc_potential(nuc_x,nuc_charge):
    n_nuc = len(nuc_x)
    x_ = np.expand_dims(nuc_x, axis=0)
    nuc_xs_1 = np.tile(x_,(n_nuc,1,1))
    nuc_xs_2 = np.tile(np.transpose(x_,(1,0,2)),(1,n_nuc,1))
    diff = nuc_xs_1 - nuc_xs_2
    norms = np.linalg.norm(diff,axis=(2))
    norms_inv = 1/(norms+np.eye(n_nuc))-np.eye(n_nuc)
    #print(norms_inv)
    _charge = np.expand_dims(nuc_charge, axis=0)
    norms_inv_charged = norms_inv.transpose()*(_charge.T@_charge)
    #print(norms_inv_charged)
    v_nuc_nuc = np.sum(norms_inv_charged,axis=(0,1))/2
    return v_nuc_nuc

if __name__ == "__main__":
    n_electron = 3
    x = np.array([1,0,0,3,0,0,4,0,0])
    nuc_x = np.array([[0,0,0]])
    nuc_charge = np.array([2])
    #v_ele_ele = get_ele_ele_potential(x)
    nuc_ele_potential = get_nuc_ele_potential(nuc_x,nuc_charge,n_electron)
    v_nuc_ele = nuc_ele_potential(x)
    print(v_nuc_ele)
    v_nuc_nuc = nuc_nuc_potential(nuc_x, nuc_charge)
    print(v_nuc_nuc)
    v_ele_ele = ele_ele_potential(x)
    print(v_ele_ele)
    

