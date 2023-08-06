"""Toy code implementing the time evolving block decimation (TEBD)."""

import numpy as np
from scipy.linalg import expm
from a_mps import split_truncate_theta
import tfi_exact


def calc_U_bonds(model, dt):
    """Given a model, calculate ``U_bonds[i] = expm(-dt*model.H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means imaginary time evolution!
    """
    H_bonds = model.H_bonds
    d = H_bonds[0].shape[0]
    U_bonds = []
    for H in H_bonds:
        H = np.reshape(H, [d * d, d * d])
        U = expm(-dt * H)
        U_bonds.append(np.reshape(U, [d, d, d, d]))
    return U_bonds


def run_TEBD(psi, U_bonds, model, E_exact, N_steps, chi_max, eps):
    """Evolve the state `psi` for `N_steps` time steps with (first order) TEBD.

    The state psi is modified in place."""
    Nbonds = psi.L - 1

    errors = []

    assert len(U_bonds) == Nbonds
    for n in range(N_steps): #this are the number of time steps perfomed for the dt that gave rise to U_bonds
        for k in [0, 1]:  # even, odd
            for i_bond in range(k, Nbonds, 2):
                update_bond(psi, i_bond, U_bonds[i_bond], chi_max, eps)
        E = model.energy(psi)
        errors.append(abs((E - E_exact) / E_exact))

    assert len(errors) == N_steps
    return errors, E
    # done


def update_bond(psi, i, U_bond, chi_max, eps):
    """Apply `U_bond` acting on i,j=(i+1) to `psi`."""
    j = i + 1
    # construct theta matrix
    theta = psi.get_theta2(i)  # vL i j vR
    # apply U
    Utheta = np.tensordot(U_bond, theta, axes=([2, 3], [1, 2]))  # i j [i*] [j*], vL [i] [j] vR
    Utheta = np.transpose(Utheta, [2, 0, 1, 3])  # vL i j vR
    # split and truncate
    Ai, Sj, Bj = split_truncate_theta(Utheta, chi_max, eps)
    # put back into MPS
    Gi = np.tensordot(np.diag(psi.Ss[i]**(-1)), Ai, axes=[1, 0])  # vL [vL*], [vL] i vC
    psi.Bs[i] = np.tensordot(Gi, np.diag(Sj), axes=[2, 0])  # vL i [vC], [vC] vC
    psi.Ss[j] = Sj  # vC
    psi.Bs[j] = Bj  # vC j vR



def TEBD_gs_finite(L, J, g, E_exact='',iterations=100, dts_len = 5):

    """
    calculate the imaginary time evolution to find the gs of the hamiltonian given by
    L, J and g.
    dts_len: determines how many of the dt in dts list will be used
    iterations should be total number of iterations
    """
    assert 1<= dts_len <= 5, 'len of dts chosen should be between 1 and 5'

    if not E_exact:
        E_exact = tfi_exact.finite_gs_energy(L, J, g)
    print("finite TEBD, (imaginary time evolution)")
    print("L={L:d}, J={J:.1f}, g={g:.2f}".format(L=L, J=J, g=g))
    import a_mps
    import b_model
    model = b_model.TFIModel(L, J=J, g=g)
    psi = a_mps.init_spinup_MPS(L)
    errors = []
    dts = [0.1, 0.01, 0.001, 1.e-4, 1.e-5]
    N_steps = iterations//dts_len #how many steps for each of the dt will be used
    for dt in dts[:dts_len]:
        U_bonds = calc_U_bonds(model, dt)
        err, E = run_TEBD(psi, U_bonds, model, E_exact, N_steps=N_steps, chi_max=30, eps=1.e-10)
        errors += err
        print("dt = {dt:.5f}: E = {E:.13f}".format(dt=dt, E=E))
    print("final bond dimensions: ", psi.get_chi())
    # if L < 20:  # for small systems compare to exact diagonalization
    #     E_exact = tfi_exact.finite_gs_energy(L, 1., g)
    #     print("Exact diagonalization: E = {E:.13f}".format(E=E_exact))
    #     print("relative error: ", abs((E - E_exact) / E_exact))
    return errors, psi


if __name__ == "__main__":
    TEBD_gs_finite(L=14, J=1., g=1.5)
