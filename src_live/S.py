import numpy as np 
from numba import njit
@njit(cache = True, fastmath = True)
def S(Dx, Dy, Dz, P, Q, RAB2, alpha, beta):
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 0):
        return np.pi**(3/2)*np.exp(-Q*RAB2/P)/P**(3/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 0, 0):
        return np.pi**(3/2)*Dx*alpha*np.exp(-Q*RAB2/P)/P**(5/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 1, 0):
        return np.pi**(3/2)*Dy*alpha*np.exp(-Q*RAB2/P)/P**(5/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 1):
        return np.pi**(3/2)*Dz*alpha*np.exp(-Q*RAB2/P)/P**(5/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*(P**2 + alpha*(2*Dx**2*Q - P))*np.exp(-Q*RAB2/P)/(P**(7/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 1, 0):
        return np.pi**(3/2)*Dx*Dy*alpha**2*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 0, 1):
        return np.pi**(3/2)*Dx*Dz*alpha**2*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*(P**2 + alpha*(2*Dy**2*Q - P))*np.exp(-Q*RAB2/P)/(P**(7/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 1, 1):
        return np.pi**(3/2)*Dy*Dz*alpha**2*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*(P**2 + alpha*(2*Dz**2*Q - P))*np.exp(-Q*RAB2/P)/(P**(7/2)*beta)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 0):
        return -np.pi**(3/2)*Dx*beta*np.exp(-Q*RAB2/P)/P**(5/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 0, 0):
        return np.pi**(3/2)*(-Dx**2*Q + (1/2)*P)*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 1, 0):
        return -np.pi**(3/2)*Dx*Dy*Q*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 1):
        return -np.pi**(3/2)*Dx*Dz*Q*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 2, 0, 0):
        return -1/2*np.pi**(3/2)*Dx*(P**2 - alpha*(-2*Dx**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*Dy*alpha*(-2*Dx**2*Q + P)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*Dz*alpha*(-2*Dx**2*Q + P)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*Dx*(-P**2 - alpha*(2*Dy**2*Q - P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 1, 1):
        return -np.pi**(3/2)*Dx*Dy*Dz*Q*alpha*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*Dx*(-P**2 - alpha*(2*Dz**2*Q - P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 0):
        return -np.pi**(3/2)*Dy*beta*np.exp(-Q*RAB2/P)/P**(5/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 0, 0):
        return -np.pi**(3/2)*Dx*Dy*Q*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 1, 0):
        return np.pi**(3/2)*(-Dy**2*Q + (1/2)*P)*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 1):
        return -np.pi**(3/2)*Dy*Dz*Q*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*Dy*(-P**2 - alpha*(2*Dx**2*Q - P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*Dx*alpha*(-2*Dy**2*Q + P)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 0, 1):
        return -np.pi**(3/2)*Dx*Dy*Dz*Q*alpha*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 2, 0):
        return -1/2*np.pi**(3/2)*Dy*(P**2 - alpha*(-2*Dy**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*Dz*alpha*(-2*Dy**2*Q + P)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*Dy*(-P**2 - alpha*(2*Dz**2*Q - P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 0):
        return -np.pi**(3/2)*Dz*beta*np.exp(-Q*RAB2/P)/P**(5/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 0, 0):
        return -np.pi**(3/2)*Dx*Dz*Q*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 1, 0):
        return -np.pi**(3/2)*Dy*Dz*Q*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 1):
        return np.pi**(3/2)*(-Dz**2*Q + (1/2)*P)*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*Dz*(-P**2 - alpha*(2*Dx**2*Q - P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 1, 0):
        return -np.pi**(3/2)*Dx*Dy*Dz*Q*alpha*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*Dx*alpha*(-2*Dz**2*Q + P)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*Dz*(-P**2 - alpha*(2*Dy**2*Q - P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*Dy*alpha*(-2*Dz**2*Q + P)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 2):
        return -1/2*np.pi**(3/2)*Dz*(P**2 - alpha*(-2*Dz**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 0, 0):
        return (1/2)*np.pi**(3/2)*(P**2 + beta*(2*Dx**2*Q - P))*np.exp(-Q*RAB2/P)/(P**(7/2)*alpha)
    if (i, j, k, l, m, n) == (2, 0, 0, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*Dx*(P**2 + beta*(2*Dx**2*Q - 3*P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*Dy*(P**2 + beta*(2*Dx**2*Q - P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*Dz*(P**2 + beta*(2*Dx**2*Q - P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 2, 0, 0):
        return (1/4)*np.pi**(3/2)*(P**4 + P**2*alpha*(2*Dx**2*Q - P) + P**2*beta*(2*Dx**2*Q - P) + Q*(4*Dx**4*Q**2 - 12*Dx**2*P*Q + 3*P**2))*np.exp(-Q*RAB2/P)/(P**(11/2)*Q)
    if (i, j, k, l, m, n) == (2, 0, 0, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dy*alpha*(P**2 - beta*(-2*Dx**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dz*alpha*(P**2 - beta*(-2*Dx**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 2, 0):
        return (1/4)*np.pi**(3/2)*(P**4 + P**2*alpha*(2*Dy**2*Q - P) + P**2*beta*(2*Dx**2*Q - P) + Q*(4*Dx**2*Dy**2*Q**2 + P**2 - 2*P*Q*(Dx**2 + Dy**2)))*np.exp(-Q*RAB2/P)/(P**(11/2)*Q)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*Dy*Dz*alpha*(P**2 + beta*(2*Dx**2*Q - P))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 0, 2):
        return (1/4)*np.pi**(3/2)*(P**4 + P**2*alpha*(2*Dz**2*Q - P) + P**2*beta*(2*Dx**2*Q - P) + Q*(4*Dx**2*Dz**2*Q**2 + P**2 - 2*P*Q*(Dx**2 + Dz**2)))*np.exp(-Q*RAB2/P)/(P**(11/2)*Q)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 0, 0):
        return np.pi**(3/2)*Dx*Dy*beta**2*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*Dy*beta*(2*Dx**2*Q - P)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*Dx*beta*(2*Dy**2*Q - P)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 0, 1):
        return np.pi**(3/2)*Dx*Dy*Dz*Q*beta*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dy*beta*(P**2 - alpha*(-2*Dx**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 1, 1, 0):
        return np.pi**(3/2)*(Dx**2*Dy**2*Q**2 + (1/4)*P**2 - 1/2*P*Q*(Dx**2 + Dy**2))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*Dy*Dz*Q*(2*Dx**2*Q - P)*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dy*beta*(P**2 - alpha*(-2*Dy**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dz*Q*(2*Dy**2*Q - P)*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*Dx*Dy*beta*(P**2 + alpha*(2*Dz**2*Q - P))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 0, 0):
        return np.pi**(3/2)*Dx*Dz*beta**2*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*Dz*beta*(2*Dx**2*Q - P)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 1, 0):
        return np.pi**(3/2)*Dx*Dy*Dz*Q*beta*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*Dx*beta*(2*Dz**2*Q - P)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dz*beta*(P**2 - alpha*(-2*Dx**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*Dy*Dz*Q*(2*Dx**2*Q - P)*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 1, 0, 1):
        return np.pi**(3/2)*(Dx**2*Dz**2*Q**2 + (1/4)*P**2 - 1/2*P*Q*(Dx**2 + Dz**2))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dz*beta*(P**2 + alpha*(2*Dy**2*Q - P))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dy*Q*(2*Dz**2*Q - P)*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*Dx*Dz*beta*(P**2 - alpha*(-2*Dz**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 0, 0):
        return (1/2)*np.pi**(3/2)*(P**2 + beta*(2*Dy**2*Q - P))*np.exp(-Q*RAB2/P)/(P**(7/2)*alpha)
    if (i, j, k, l, m, n) == (0, 2, 0, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*Dx*(P**2 + beta*(2*Dy**2*Q - P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*Dy*(P**2 + beta*(2*Dy**2*Q - 3*P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*Dz*(P**2 + beta*(2*Dy**2*Q - P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 2, 0, 0):
        return (1/4)*np.pi**(3/2)*(P**4 + P**2*alpha*(2*Dx**2*Q - P) + P**2*beta*(2*Dy**2*Q - P) + Q*(4*Dx**2*Dy**2*Q**2 + P**2 - 2*P*Q*(Dx**2 + Dy**2)))*np.exp(-Q*RAB2/P)/(P**(11/2)*Q)
    if (i, j, k, l, m, n) == (0, 2, 0, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dy*alpha*(P**2 - beta*(-2*Dy**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dz*alpha*(P**2 + beta*(2*Dy**2*Q - P))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 2, 0):
        return (1/4)*np.pi**(3/2)*(P**4 + P**2*alpha*(2*Dy**2*Q - P) + P**2*beta*(2*Dy**2*Q - P) + Q*(4*Dy**4*Q**2 - 12*Dy**2*P*Q + 3*P**2))*np.exp(-Q*RAB2/P)/(P**(11/2)*Q)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*Dy*Dz*alpha*(P**2 - beta*(-2*Dy**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 0, 2):
        return (1/4)*np.pi**(3/2)*(P**4 + P**2*alpha*(2*Dz**2*Q - P) + P**2*beta*(2*Dy**2*Q - P) + Q*(4*Dy**2*Dz**2*Q**2 + P**2 - 2*P*Q*(Dy**2 + Dz**2)))*np.exp(-Q*RAB2/P)/(P**(11/2)*Q)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 0, 0):
        return np.pi**(3/2)*Dy*Dz*beta**2*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 1, 0, 0):
        return np.pi**(3/2)*Dx*Dy*Dz*Q*beta*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*Dz*beta*(2*Dy**2*Q - P)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*Dy*beta*(2*Dz**2*Q - P)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*Dy*Dz*beta*(P**2 + alpha*(2*Dx**2*Q - P))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dz*Q*(2*Dy**2*Q - P)*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dy*Q*(2*Dz**2*Q - P)*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*Dy*Dz*beta*(P**2 - alpha*(-2*Dy**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 1, 1):
        return np.pi**(3/2)*(Dy**2*Dz**2*Q**2 + (1/4)*P**2 - 1/2*P*Q*(Dy**2 + Dz**2))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*Dy*Dz*beta*(P**2 - alpha*(-2*Dz**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 0, 0):
        return (1/2)*np.pi**(3/2)*(P**2 + beta*(2*Dz**2*Q - P))*np.exp(-Q*RAB2/P)/(P**(7/2)*alpha)
    if (i, j, k, l, m, n) == (0, 0, 2, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*Dx*(P**2 + beta*(2*Dz**2*Q - P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*Dy*(P**2 + beta*(2*Dz**2*Q - P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*Dz*(P**2 + beta*(2*Dz**2*Q - 3*P))*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 2, 0, 0):
        return (1/4)*np.pi**(3/2)*(P**4 + P**2*alpha*(2*Dx**2*Q - P) + P**2*beta*(2*Dz**2*Q - P) + Q*(4*Dx**2*Dz**2*Q**2 + P**2 - 2*P*Q*(Dx**2 + Dz**2)))*np.exp(-Q*RAB2/P)/(P**(11/2)*Q)
    if (i, j, k, l, m, n) == (0, 0, 2, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dy*alpha*(P**2 + beta*(2*Dz**2*Q - P))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dz*alpha*(P**2 - beta*(-2*Dz**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 2, 0):
        return (1/4)*np.pi**(3/2)*(P**4 + P**2*alpha*(2*Dy**2*Q - P) + P**2*beta*(2*Dz**2*Q - P) + Q*(4*Dy**2*Dz**2*Q**2 + P**2 - 2*P*Q*(Dy**2 + Dz**2)))*np.exp(-Q*RAB2/P)/(P**(11/2)*Q)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*Dy*Dz*alpha*(P**2 - beta*(-2*Dz**2*Q + 3*alpha + 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 0, 2):
        return (1/4)*np.pi**(3/2)*(P**4 + P**2*alpha*(2*Dz**2*Q - P) + P**2*beta*(2*Dz**2*Q - P) + Q*(4*Dz**4*Q**2 - 12*Dz**2*P*Q + 3*P**2))*np.exp(-Q*RAB2/P)/(P**(11/2)*Q)