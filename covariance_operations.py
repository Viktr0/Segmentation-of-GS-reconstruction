import numpy as np
import torch


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    L = R @ L
    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    # actual_covariance = strip_symmetric(actual_covariance)
    return actual_covariance


def get_covariance(scaling, rotation, scaling_modifier=1):
    return build_covariance_from_scaling_rotation(torch.exp(scaling), scaling_modifier, rotation)


def computeCov3D(scale, rot, mod=1):

    # Modifications
    #scale = np.exp(scale)
    # rot = rot / norm[:, None]

    S = np.zeros((3, 3))
    S[0][0] = mod * scale[0]
    S[1][1] = mod * scale[1]
    S[2][2] = mod * scale[2]

    r = rot[0]
    x = rot[1]
    y = rot[2]
    z = rot[3]

    R = np.zeros((3, 3))
    R[0][0] = 1.0-2.0*(y*y+z*z)
    R[0][1] = 2.0*(x*y-r*z)
    R[0][2] = 2.0*(x*z+r*y)
    R[1][0] = 2.0*(x*y+r*z)
    R[1][1] = 1.0-2.0*(x*x+z*z)
    R[1][2] = 2.0*(y*z-r*x)
    R[2][0] = 2.0*(x*z-r*y)
    R[2][1] = 2.0*(y*z+r*x)
    R[2][2] = 1.0-2.0*(x*x+y*y)

    return R @ S @ S.T @ R.T

def get_abc(cov_mat):

        # Find and sort eigenvalues to correspond to the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov_mat)
        idx = np.sum(cov_mat, axis=0).argsort()
        eigvals_temp = eigvals[idx]
        idx = eigvals_temp.argsort()
        eigvals = eigvals[idx]

        # Width, height and depth of ellipsoid
        rx, ry, rz = np.sqrt(eigvals)
        return rx, ry, rz