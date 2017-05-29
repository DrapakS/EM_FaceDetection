__author__ = 'Stepan'
import numpy as np
EPS = 1e-100
import matplotlib.pyplot as plt

def get_lpx_d_all(X, F, B, s):
##################################################################
#
# Calculates log(p(X_k|d_k,F,B,s)) for all images X_k in X and
# all possible displacements d_k.
#
# Input parameters:
#
#   X ... H x W x N numpy.array, N images of size H x W
#   F ... h x w numpy.array, estimate of villain's face
#   B ... H x W numpy.array, estimate of background
#   s ... 1 x 1, estimate of standart deviation of Gaussian noise
#
# Output parameters:
#
#   lpx_d_all ... (H-h+1) x (W-w+1) x N numpy.array,
#                 lpx_d_all[dh,dw,k] - log-likelihood of
#                 observing image X_k given that the villain's
#                 face F is located at displacement (dh, dw)
#
##################################################################
    H = X.shape[0]
    W = X.shape[1]
    h = F.shape[0]
    w = F.shape[1]
    N = X.shape[2]

    bg_lh = -(X - B[:, :, None]) ** 2
    full_bg_lh = np.sum(bg_lh, axis=(0, 1))
    total_lh = np.zeros((H - h + 1, W - w + 1, N))
    denum = 0.5 * W * H * np.log(2 * np.pi * s)
    for ch in range(H - h + 1):
        for cw in range(W - w + 1):
            c_face_lh = -np.sum((X[ch:ch + h, cw:cw + w, :] - F[:, :, None])**2, axis=(0, 1))
            c_bg_lh = -np.sum(bg_lh[ch:ch + h, cw:cw + w, :], axis=(0, 1))
            total_lh[ch, cw, :] = c_face_lh + c_bg_lh + full_bg_lh
    return 0.5 * total_lh / s - denum



def calc_L(X, F, B, s, A, q, useMAP = False):
###################################################################
#
# Calculates the lower bound L(q,F,B,s,A) for the marginal log
# likelihood
#
# Input parameters:
#
#   X ... H x W x N numpy.array, N images of size H x W
#   F ... h x w numpy.array, estimate of villain's face
#   B ... H x W numpy.array, estimate of background
#   s ... 1 x 1, estimate of standart deviation of Gaussian noise
#   A ... (H-h+1) x (W-w+1) numpy.array, estimate of prior on
#         displacement of face in any image
#   q  ... if useMAP = False:
#             (H-h+1) x (W-w+1) x N numpy.array,
#             q[dh,dw,k] - estimate of posterior of displacement
#             (dh,dw) of villain's face given image Xk
#           if useMAP = True:
#             2 x N numpy.array,
#             q[0,k] - MAP estimates of dh for X_k
#             q[1,k] - MAP estimates of dw for X_k
#   useMAP ... logical, if true then q is a MAP estimates of
#              displacement (dh,dw) of villain's face given image
#              Xk
#
# Output parameters:
#
#   L ... 1 x 1, the lower bound L(q,F,B,s,A) for the marginal log
#         likelihood
#
###################################################################

    lpx_d = get_lpx_d_all(X, F, B, s)
    if useMAP:
        l = np.sum(np.log(A[q[0], q[1]] + EPS) + lpx_d[q[0], q[1], range(X.shape[2])])
    else:
        l = np.sum(q * (-np.log(q + EPS) + np.log(A[:, :, None] + EPS) + lpx_d))
    return l


def e_step(X, F, B, s, A, useMAP = False):
##################################################################
#
# Given the current esitmate of the parameters, for each image Xk
# esitmates the probability p(d_k|X_k,F,B,s,A)
#
# Input parameters:
#
#   X ... H x W x N numpy.array, N images of size H x W
#   F ... h x w numpy.array, estimate of villain's face
#   B ... H x W numpy.array, estimate of background
#   s ... 1 x 1, estimate of standart deviation of Gaussian noise
#   A ... (H-h+1) x (W-w+1) numpy.array, estimate of prior on
#         displacement of face in any image
#   useMAP ... logical, if true then q is a MAP estimates of
#              displacement (dh,dw) of villain's face given image
#              Xk
#
# Output parameters:
#
#   q  ... if useMAP = False:
#             (H-h+1) x (W-w+1) x N numpy.array,
#             q[dh,dw,k] - estimate of posterior of displacement
#             (dh,dw) of villain's face given image Xk
#           if useMAP = True:
#             2 x N numpy.array,
#             q[0,k] - MAP estimates of dh for X_k
#             q[1,k] - MAP estimates of dw for X_k
###################################################################

    lpx_d = get_lpx_d_all(X, F, B, s)
    if useMAP:
        start_shape = lpx_d.shape[:2]
        lpx_d = np.reshape(lpx_d, (-1, X.shape[2]))
        max_ind = np.argmax(lpx_d, axis=0)
        q1, q2 = np.unravel_index(max_ind, start_shape)
        q = np.vstack((q1, q2))
    else:
        q = lpx_d + np.log(A[:, :, None] + EPS)
        q -= np.max(q, axis=(0, 1))[None, None, :]
        q = np.exp(q)
        q /= np.sum(q, axis=(0, 1))
    return q


def m_step(X, q, h, w, useMAP = False):
###################################################################
#
# Estimates F,B,s,A given esitmate of posteriors defined by q
#
# Input parameters:
#
#   X ... H x W x N numpy.array, N images of size H x W
#   q ... if useMAP = False:
#             (H-h+1) x (W-w+1) x N numpy.array,
#             q[dh,dw,k] - estimate of posterior of displacement
#             (dh,dw) of villain's face given image Xk
#           if useMAP = True:
#             2 x N numpy.array,
#             q[0,k] - MAP estimates of dh for X_k
#             q[1,k] - MAP estimates of dw for X_k
#   h ... 1 x 1, face mask hight
#   w ... 1 x 1, face mask widht
#  useMAP ... logical, if true then q is a MAP estimates of
#             displacement (dh,dw) of villain's face given image
#             Xk
#
# Output parameters:
#
#   F ... h x w numpy.array, estimate of villain's face
#   B ... H x W numpy.array, estimate of background
#   s ... 1 x 1, estimate of standart deviation of Gaussian noise
#   A ... (H-h+1) x (W-w+1) numpy.array, estimate of prior on
#         displacement of face in any image
###################################################################
    H, W = X.shape[:2]
    F = np.zeros((h, w))
    N = X.shape[2]
    if useMAP:

        A = np.zeros((H-h+1, W-w+1))
        B = np.sum(X, axis=2)
        denum = X.shape[2] * np.ones(B.shape)
        for i in range(q.shape[1]):
            ch = q[0, i]
            cw = q[1, i]
            B[ch:ch+h, cw:cw+w] -= X[ch:ch+h, cw:cw+w, i]
            #T[ch:ch+h, cw:cw+w, :] -= 1
            F += X[ch:ch+h, cw:cw+w, i].astype(float)
            A[q[0, i], q[1, i]] += 1
            denum[ch:ch+h, cw:cw+w] -= 1

        B /= denum + EPS
        F /= N
        A /= N
        s = np.sum((X - B[:, :, None]) ** 2)
        for i in range(q.shape[1]):
            ch = q[0, i]
            cw = q[1, i]
            s += np.sum((X[ch:ch+h, cw:cw+w, i] - F) ** 2) - \
                 np.sum((X[ch:ch+h, cw:cw+w, i] - B[ch:ch+h, cw:cw+w]) ** 2)
        s /= N * W * H

    else:
        for i in range(h):
            for j in range(w):
                F[i, j] = np.sum(q * X[i:i + H - h + 1, j:j + W - w + 1, :])
        F /= N

        T = np.ones(X.shape, dtype=float)
        for i in range(H - h + 1):
            for j in range(W - w + 1):
                T[i:i + h, j:j + w, :] -= q[i, j, :][None, None, :]
        B = np.sum(X * T, axis=2)/np.sum(T, axis=2)

        s = np.sum(T * (X - B[:, :, None]) ** 2)
        for i in range(H - h + 1):
            for j in range(W - w + 1):
                s += np.sum(q[i, j, :][None, None, :] * (X[i:i + h, j:j + w, :] - F[:, :, None]) ** 2)
        s /= N * W * H
        A = np.sum(q, axis=2) / N

    return F, B, s, A


def run_EM(X, h, w, F=None, B = None, s = None, A = None,
    tolerance = 0.001, max_iter = 50, useMAP = False):
###################################################################
#
# Runs EM loop until the likelihood of observing X given current
# estimate of parameters is idempotent as defined by a fixed
# tolerance
#
# Input parameters:
#
#   X ... H x W x N numpy.array, N images of size H x W
#   h ... 1 x 1, face mask hight
#   w ... 1 x 1, face mask widht
#   F, B, s, A ... initial parameters (optional!)
#   F ... h x w numpy.array, estimate of villain's face
#   B ... H x W numpy.array, estimate of background
#   s ... 1 x 1, estimate of standart deviation of Gaussian noise
#   A ... (H-h+1) x (W-w+1) numpy.array, estimate of prior on
#         displacement of face in any image
#   tolerance ... parameter for stopping criterion
#   max_iter  ... maximum number of iterations
#   useMAP ... logical, if true then after E-step we take only
#              MAP estimates of displacement (dh,dw) of villain's
#              face given image Xk
#
#
# Output parameters:
#
#   F, B, s, A ... trained parameters
#   LL ... 1 x (number_of_iters + 2) numpy.array, L(q,F,B,s,A)
#          at initial guess, after each EM iteration and after
#          final estimate of posteriors;
#          number_of_iters is actual number of iterations that was
#          done
###################################################################

    if F is None:
        F = np.random.randint(low=50, high=200, size=(h, w))
    if B is None:
        B = np.random.randint(low=50, high=200, size=(X.shape[:2]))
    if s is None:
        s = 500
    if A is None:
        A = np.random.random(size=(X.shape[0]-h+1, X.shape[1]-w+1))

    LL = []
    for i in range(max_iter):
        q = e_step(X, F, B, s, A, useMAP)
        F, B, s, A = m_step(X, q, h, w, useMAP)
        llh = calc_L(X, F, B, s, A, q, useMAP)
        LL.append(llh)
        print llh
        if i > 0:
            print '========='
            print LL[-1]
            print LL[-2]
            print LL[-1] - LL[-2]
            print '========='
            if LL[-1] - LL[-2] < tolerance:
                break
        if i % 5 == 0:
            plt.imshow(F, cmap='gray')
            plt.axis('off')
            plt.show()
            plt.imshow(B, cmap='gray')
            plt.axis('off')
            plt.show()
    plt.imshow(F, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.imshow(B, cmap='gray')
    plt.axis('off')
    plt.show()
    return F, B, s, A, LL


def run_EM_with_restarts(X, h, w, tolerance = 0.001, max_iter = 50,
                     useMAP = False, restart=10):
###################################################################
#
# Restarts EM several times from different random initializations
# and stores the best estimate of the parameters as measured by
# the L(q,F,B,s,A)
#
# Input parameters:
#
#   X ... H x W x N numpy.array, N images of size H x W
#   h ... 1 x 1, face mask hight
#   w ... 1 x 1, face mask widht
#   tolerance, max_iter, useMAP ... parameters for EM
#   restart   ... number of EM runs
#
# Output parameters:
#
#   F ... h x w numpy.array, the best estimate of villain's face
#   B ... H x W numpy.array, the best estimate of background
#   s ... 1 x 1, the best estimate of standart deviation of
#         Gaussian noise
#   A ... (H-h+1) x (W-w+1) numpy.array, the best estimate of
#         prior on displacement of face in any image
#   LL ... 1 x 1, the best L(q,F,B,s,A)
###################################################################

        best_score = -np.inf
        for i in range(restart):
            cF, cB, cs, cA, llh_list = run_EM(X, h, w, tolerance=tolerance, max_iter=max_iter, useMAP=useMAP)
            if llh_list[-1] > best_score:
                F = cF
                B = cB
                s = cs
                A = cA
                LL = llh_list
        return F, B, s, A, LL[-1]
