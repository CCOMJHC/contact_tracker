#!/usr/bin/env python

import numpy as np
import scipy.spatial.transform.rotation as R
from numba import jit

class EulerRotationWithUncertainty(R.Rotation):

    def __init__(self,quat, normalized=False,copy=False):
        R.Rotation.__init__(self,quat,normalized=False,copy=False)
        self.angles = None
        self.seq = None
        self.degrees = None
        self.Crpy = None

    @classmethod
    def from_eulerWithUncertainty(cls, seq, angles, Crpy, degrees=False,):
        C = cls.from_euler(seq, angles, degrees)
        C.seq = seq
        C.angles = angles
        C.degrees = degrees
        if degrees:
            C.Crpy = np.deg2rad(Crpy)
        else:
            C.Crpy = Crpy

        return C

    def applyWithUncertainty(self,X,Cxyz):

        h,p,r = self.as_euler(self.seq)

        x = X[0]
        y = X[1]
        z = X[2]

        sin_r = np.sin(r)
        cos_r = np.cos(r)
        sin_p = np.sin(p)
        cos_p = np.cos(p)
        sin_h = np.sin(h)
        cos_h = np.cos(h)
        M = self.as_dcm()  # Returns rotation matrix.

        # Build the covariance
        C = np.zeros((6,6))
        C[:3,:3] = self.Crpy[:,:]
        C[3:,3:] = Cxyz[:,:]

        # Build the Jacobian
        J = np.zeros((3,6))
        J[0,0] = sin_r * (y * sin_h - z * cos_h * sin_p) + cos_r * (y * cos_h * sin_p + z * sin_h)
        J[0,1] = cos_h * (z * cos_r * cos_p + y * sin_r * cos_p - x * sin_p)
        J[0,2] = z * cos_h * sin_r - (x * cos_p + y * sin_r * sin_p) * sin_h - cos_r * (y * cos_h + z * sin_p * sin_h)
        #J[0,3] = M[0,0]
        #J[0,4] = M[0,1]
        #J(1,6,:,:) = M(1,3,:,:);
        J[1,0] = cos_r * (y * sin_p * sin_h - z * cos_h) - sin_r * (y * cos_h + z * sin_p * sin_h)
        J[1,1] = (z * cos_r * cos_p + y * sin_r * cos_p - x * sin_p) * sin_h
        J[1,2] = x * cos_p * cos_h + cos_r * (z * cos_h * sin_p - y * sin_h) + sin_r * (y * cos_h * sin_p + z * sin_h)
        #J(2,4,:,:) = M(2,1,:,:)
        #J(2,5,:,:) = M(2,2,:,:)
        #J(2,6,:,:) = M(2,3,:,:)
        J[2,0] = cos_p * (y * cos_r - z * sin_r)
        J[2,1] = -x * cos_p - (z * cos_r + y * sin_r) * sin_p
        J[2,2] = 0.0
        #J(3,4,:,:) = M(3,1,:,:);
        #J(3,5,:,:) = M(3,2,:,:);
        #J(3,6,:,:) = M(3,3,:,:);
        J[:,3:] = M[:,:]

        # Do the rotation.
        Xout = self.apply(X)
        # Propagate the uncertainty.
        Cout = np.matmul(J,np.matmul(C,J.T))
        C = C[None,:,:]
        J = J[None,:,:]
        #Cout = np.einsum('ijk','ik->ij',J,np.einsum('ijk','ik->ij',C,J.T))
        return [Xout, Cout]

def rotate_with_uncertainty(RPY,Crpy,XYZ,Cxyz,degrees=False):
    Robj = R.Rotation.from_euler('zyx', RPY, degrees=degrees)
    M = Robj.as_dcm()  # Returns rotation matrix.
    return _rotate_with_uncertainty(M,RPY,Crpy,XYZ,Cxyz)

@jit(nopython = Trcacheue, nogil = True, parallel=True)
def _rotate_with_uncertainty(M,RPY,Crpy,XYZ,Cxyz):

    # Simplify some complex math.
    r = RPY[0]
    p = RPY[1]
    h = RPY[2]

    sin_r = np.sin(r)
    cos_r = np.cos(r)
    sin_p = np.sin(p)
    cos_p = np.cos(p)
    sin_h = np.sin(h)
    cos_h = np.cos(h)

    x = XYZ[0]
    y = XYZ[1]
    z = XYZ[2]

    # Build the covariance
    C = np.zeros((6, 6))
    C[:3, :3] = Crpy[:, :]
    C[3:, 3:] = Cxyz[:, :]

    # Build the Jacobian
    J = np.zeros((3, 6))
    J[0, 0] = sin_r * (y * sin_h - z * cos_h * sin_p) + cos_r * (y * cos_h * sin_p + z * sin_h)
    J[0, 1] = cos_h * (z * cos_r * cos_p + y * sin_r * cos_p - x * sin_p)
    J[0, 2] = z * cos_h * sin_r - (x * cos_p + y * sin_r * sin_p) * sin_h - cos_r * (y * cos_h + z * sin_p * sin_h)
    J[1, 0] = cos_r * (y * sin_p * sin_h - z * cos_h) - sin_r * (y * cos_h + z * sin_p * sin_h)
    J[1, 1] = (z * cos_r * cos_p + y * sin_r * cos_p - x * sin_p) * sin_h
    J[1, 2] = x * cos_p * cos_h + cos_r * (z * cos_h * sin_p - y * sin_h) + sin_r * (y * cos_h * sin_p + z * sin_h)
    J[2, 0] = cos_p * (y * cos_r - z * sin_r)
    J[2, 1] = -x * cos_p - (z * cos_r + y * sin_r) * sin_p
    J[2, 2] = 0.0
    J[:, 3:] = M[:, :]

    # Do the rotation.
    XYZout = np.dot(M,XYZ)
    #Xout = Robj.apply(X)
    # Propagate the uncertainty.
    Cout = np.dot(J,np.dot(C,J.T))

    return (XYZout, Cout)