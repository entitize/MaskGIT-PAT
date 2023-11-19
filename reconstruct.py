import os
import numpy as np
import pickle
from scipy import signal
from scipy.fft import fft, ifft
from numpy.lib.twodim_base import diag
from matplotlib import pyplot as plt
import scipy
import json
import argparse

def isInRange(theta, theta1, theta2):
  # This function computes the logical value of whether theta lies inside the
  # arc defined by counter clockwise rotating from theta1 to theta2.

  return ((theta - theta1) % (2 * np.pi)) <= ((theta2 - theta1) % (2 * np.pi))

def intersectHlineAndCircle(x_r, y_r, R, y_line):
  # This function computes the coordinate of intersection points of a circle
  # centered at x_r, y_r with a radius of R and a horizontal line whose y
  # coordinate is y_line.

  if abs(y_r - y_line) >= R: # if there is no intersection point
    # tangency is not counted as intersection
    x = np.zeros(0) # empty array
    y = np.zeros(0)
    theta = np.zeros(0)
    return (x, y, theta)
  else:
    theta1 = np.arcsin((y_line - y_r) / R)
    theta2 = np.pi - theta1
    y = np.hstack((y_line, y_line))

    theta = np.hstack((theta1, theta2))
    x = x_r + R * np.cos(theta)
    return (x, y, theta)

def intersectVlineAndCircle(x_r, y_r, R, x_line):
  # This function computes the coordinate of intersection points of a circle
  # centered at x_r, y_r with a radius of R and a verytical line whose x
  # coordinate is x_line.

  if abs(x_r - x_line) >= R:
    # tangency is not counted as intersection
    x = np.zeros(0)
    y = np.zeros(0)
    theta = np.zeros(0)
    return (x, y, theta)
  else:
    theta1 = np.arccos((x_line - x_r) / R)
    theta2 = - theta1
    x =  np.hstack((x_line, x_line))

    theta = np.hstack((theta1, theta2))
    y = y_r + R * np.sin(theta)
    return (x, y, theta)

def angleRangeCDMMI(x1, x2, y1, y2, x_r, y_r, R):
# This function calculates the range of the angle on the circle centered at
# x_r, y_r with a radius of R that lies inside the rectangle whose x and y
# limites are given by x1, x2, y1 and y2. The returned value is either an array
# of size 4 if the range contains two intervals, or an array of size 2 if the
# range contains one interval. If the circle does not intersect with the
# rectangle then return an empty array.

  (xh1, t, thetah1) = intersectHlineAndCircle(x_r, y_r, R, y1)
  (xh2, t, thetah2) = intersectHlineAndCircle(x_r, y_r, R, y2)
  xh      = np.hstack((xh1, xh2))
  thetah  = np.hstack((thetah1, thetah2))
  idh = np.logical_and(xh <= x2, xh >= x1)
  thetah = thetah[idh]
  # Compute the intersection points on the two horizontal lines and select
  # those points between the two vertical lines.

  (t, yv1, thetav1) = intersectVlineAndCircle(x_r, y_r, R, x1)
  (t, yv2, thetav2) = intersectVlineAndCircle(x_r, y_r, R, x2)
  yv      = np.hstack((yv1,yv2))
  thetav  = np.hstack((thetav1,thetav2))
  idv = np.logical_and(yv <= y2, yv >= y1)
  thetav = thetav[idv]
  # Compute the intersection points on the two vertical lines and select
  # those points between the two horizontal lines.

  theta = np.hstack((thetah,thetav))
  # Stack the radius angle of the points

  if theta.shape[0] == 0:
    theta_range = theta
    return theta_range
    # If there is no intersection point, returen an empty array

  theta = (theta - theta[0]) % (2 * np.pi) + theta[0]
  theta = np.sort(theta)
  theta = np.unique(theta)
  # Make the angles in accending order and delete repeated values

  theta_mid = theta[0] + (theta[1] - theta[0]) % (2 * np.pi) / 2
  x_mid = x_r + R * np.cos(theta_mid)
  y_mid = y_r + R * np.sin(theta_mid)
  # There could be either 4 intersection points or 2 intersection points.
  # The circle will be cut into 4 or 2 arcs. By checking whether the middle
  # point on the arc is in the rectangle we can determine which two or one arc
  # is that we want.

  if x_mid > x1 and x_mid < x2 and y_mid > y1 and y_mid < y2:
    theta_range = theta
    # If the first arc (middle point) is in the rectangle, we don't need to
    # change the order because it must be the first or the first and third arc
    # that is in the rectangle.
  else:
    if theta.shape[0] == 2:
        theta_range = np.hstack((theta[1], theta[0]))
        # If the first arc is not selected, the other half will be selected
    else:
        theta_range = np.hstack((theta[3], theta[0:3]))
        # If the first arc is not selected, the second and fourth will be selected
  return theta_range

def deltaThetaCDMMI(xs, ys, px, py, x_r, y_r, R):
  # This function calculate the angle arc values of a circle centered at x_r,
  # y_r with a radius of R being cut by grids made of vertical and horizontal
  # lines defined by xs and ys.

  x_id = np.zeros(0)
  y_id = np.zeros(0)
  delta_theta = np.zeros(0)
  # First asign empty arrays because we need to return something even if in the
  # end we find no intersections.

  theta_range = angleRangeCDMMI(xs[0], xs[-1], ys[0], ys[-1], x_r, y_r, R)
  if theta_range.shape[0] == 0:
    return (x_id, y_id, delta_theta)
  # Calculate the angle range of the circle that intersect with the rectangle
  # field of view.

  vlines_x        = xs[abs(xs - x_r) <= R]
  # These are the vertical lines that may intersect with the circle

  vlines_theta1   = np.arccos((vlines_x - x_r) / R)
  vlines_theta2   = - vlines_theta1
  vlines_theta    = np.hstack((vlines_theta1, vlines_theta2))
  # Calculate the radius angle of these intersection points

  hlines_y        = ys[abs(ys - y_r) <= R]
  # These are the horizontal lines that may intersect with the circle

  hlines_theta1   = np.arcsin((hlines_y - y_r) / R)
  hlines_theta2   = np.pi - hlines_theta1
  hlines_theta    = np.hstack((hlines_theta1, hlines_theta2))
  # Calculate the radius angle of these intersection points

  theta_comb  = np.hstack((vlines_theta, hlines_theta))
  # Combine all the intersection points together

  if theta_range.shape[0] == 2:
    is_in_range = isInRange(theta_comb, theta_range[0], theta_range[1])
    theta_comb  = theta_comb[is_in_range]
    theta_comb  = np.hstack((theta_comb, theta_range))
    # Because of some numerical errors, the edge may be in the end not included
    # here we add the edges manually
    theta_comb  = ((theta_comb - theta_range[0]) % (2 * np.pi)) + theta_range[0]
    theta_comb  = np.sort(theta_comb)
    theta_comb  = np.unique(theta_comb)
    # Select the intersection points that is inside the field of view using the
    # angle range. Then sort them in accending order and remove repeated points.

    delta_theta = theta_comb[1:] - theta_comb[:-1]
    theta_mid   = (theta_comb[1:] + theta_comb[:-1]) / 2
    # Calculate the middle points of the arcs formed between these points

    x_id = np.ceil((R * np.cos(theta_mid) + x_r - xs[0]) / px) - 1
    y_id = np.ceil((R * np.sin(theta_mid) + y_r - ys[0]) / py) - 1
    # From the middle point coordinates calculate the pixel index that arc lies
    # in the grids by rounding.

    x_id[x_id < 0] = 0
    x_id[x_id >= xs.shape[0]-1] = xs.shape[0] - 2
    y_id[y_id < 0] = 0
    y_id[y_id >= ys.shape[0]-1] = ys.shape[0] - 2
    # In theory the integers will not go outside the range. But because of some
    # numerical error, this can happen and we need to fix it.

  elif theta_range.shape[0] == 4:
    is_in_range1    = isInRange(theta_comb, theta_range[0], theta_range[1])
    is_in_range2    = isInRange(theta_comb, theta_range[2], theta_range[3])
    theta_comb1 = theta_comb[is_in_range1]
    theta_comb1 = np.hstack((theta_comb1, theta_range[:2]))
    # Because of some numerical errors, the edge may be in the end not included
    # here we add the edges manually
    theta_comb2 = theta_comb[is_in_range2]
    theta_comb2 = np.hstack((theta_comb2, theta_range[2:]))
    # Because of some numerical errors, the edge may be in the end not included
    # here we add the edges manually
    theta_comb1  = (theta_comb1 - theta_range[0]) % (2 * np.pi) + theta_range[0]
    theta_comb1  = np.sort(theta_comb1)
    theta_comb1  = np.unique(theta_comb1)
    theta_comb2  = (theta_comb2 - theta_range[2]) % (2 * np.pi) + theta_range[2]
    theta_comb2  = np.sort(theta_comb2)
    theta_comb2  = np.unique(theta_comb2)
    # Select the intersection points that is inside the field of view using the
    # angle range. Then sort them in accending order and remove repeated points.

    delta_theta1 = theta_comb1[1:] - theta_comb1[:-1]
    theta_mid1   = (theta_comb1[1:] + theta_comb1[:-1]) / 2
    delta_theta2 = theta_comb2[1:] - theta_comb2[:-1]
    theta_mid2   = (theta_comb2[1:] + theta_comb2[:-1]) / 2
    theta_mid   = np.hstack((theta_mid1, theta_mid2))
    delta_theta = np.hstack((delta_theta1, delta_theta2))
    # Calculate the middle points of the arcs formed between these points

    x_id = np.ceil((R * np.cos(theta_mid) + x_r - xs[0]) / px) - 1
    y_id = np.ceil((R * np.sin(theta_mid) + y_r - ys[0]) / py) - 1
    # From the middle point coordinates calculate the pixel index that arc lies
    # in the grids by rounding.

    x_id[x_id < 0] = 0
    x_id[x_id >= xs.shape[0]-1] = xs.shape[0] - 2
    y_id[y_id < 0] = 0
    y_id[y_id >= ys.shape[0]-1] = ys.shape[0] - 2
    # In theory the integers will not go outside the range. But because of some
    # numerical error, this can happen and we need to fix it.

  return (x_id, y_id, delta_theta)

def forwardMatrixFullRingCDMMI(N_transducer, R_ring, px, py, M, N, dt, N_sample, oversample, padsample, T_min, center_freq, fwhm, V_sound):
# This function computes the forward matrix of ring array photoacoustic imaging
# given the parameters of the system.
# N_transducer
#             the number of transducers evenly distributed on the array
# R_ring
#             the radius of the array [m]
# px and py
#             the pixel size in x and y direction [m]
# M and N
#             the pixel number in y and x direction
# dt
#             the time sampling interval [s]
# N_sample
#             the number of time domain samples
# oversample
#             the oversample factor, we first will first have
#             2 * oversample * N_sample points then filter and downsample
#             to get the N_sample points
# padsample
#             we also compute padsample more samples before and after the
#             2 * oversample * N_sample points before filtering and downsampling
# T_min
#             the time of the first sampling point (time zero is the laser shot)
# center_freq and fwhm
#             the center frequency and the full width half maximum of the transducer frequency response
# V_sound
#             the speed of sound [m/s]
# The function will return the forward matrix A and the sample time serie
#  t_sample and also the pixel coordinate x_sample and y_sample

  T_max = T_min + (N_sample - 1) * dt
  t_sample = dt * np.linspace(T_min, T_max, num = N_sample)

  delta_angle = 2 * np.pi / N_transducer
  angle_transducer = delta_angle * np.linspace(1, N_transducer, num = N_transducer)

  transducer_x = R_ring * np.sin(angle_transducer)
  transducer_y = R_ring * np.cos(angle_transducer)
  # Compute the transducer coordinate, they are evenly distributed on a ring

  xs = np.linspace(0, N, num = N + 1) * px - px * N / 2
  ys = np.linspace(0, M, num = M + 1) * py - py * M / 2
  x_sample = xs[:-1] + px / 2
  y_sample = ys[:-1] + py / 2
  # Compute the gird boundaries xs and xy and from which compute the pixel center coordinates

  dt_oversample = dt / 2 / oversample
  t_over_sample = np.linspace(T_min - dt_oversample  * (oversample + padsample - 0.5),\
                              T_max + dt_oversample  * (oversample + padsample - 0.5),\
                              N_sample * 2 * oversample + 2 * padsample)

  print([M, N, N_transducer, t_over_sample.shape[0]])

  A = np.zeros([M, N, N_transducer, t_sample.shape[0]])
  # Allocate space for the matrix


  N_oversample = t_over_sample.shape[0] - 1
  # After taking the derivative, the signal will become this length

  # Now generate the gaussian bandpass filter
  N_half_oversample = int(np.floor(N_oversample / 2))
  fft_freq = np.linspace(0, N_half_oversample - 1, N_half_oversample) * 1 / dt_oversample / N_oversample
  half_filter = np.exp(-(fft_freq - center_freq)**2 / ((fwhm / 2)**2) *np.log(2))
  filter = np.zeros(N_oversample)
  filter[0:N_half_oversample] = half_filter
  filter[-N_half_oversample:] = np.flip(half_filter)

  for r in range(0, N_transducer):
    A_over_sample = np.zeros([M, N, t_over_sample.shape[0]])
    for t in range(0, t_over_sample.shape[0]):
      [x_id, y_id, delta_theta] = deltaThetaCDMMI(xs, ys, px, py, transducer_x[r], transducer_y[r], t_over_sample[t] * V_sound)
      for k in range(0, delta_theta.shape[0]):
        A_over_sample[int(y_id[k]), int(x_id[k]), t] = A_over_sample[int(y_id[k]), int(x_id[k]), t] + delta_theta[k]

    A_over_sample = A_over_sample[:, :, 1:] - A_over_sample[:, :, :-1]
    # Then take the derivative operation
    # Lowpass filtering using hanning window
    A_over_sample = fft(A_over_sample, axis = -1)
    A_over_sample = A_over_sample * filter
    A_over_sample = ifft(A_over_sample, axis = -1).real
    A[:, :, [r], :] = A_over_sample[:, :, np.newaxis, oversample + padsample - 2:-padsample:2 * oversample]
    print("Building equations number %d / %d transducers" %(r + 1, N_transducer))

  A = A.reshape((M * N, N_transducer * t_sample.shape[0]))
  return (A, x_sample, y_sample, t_sample)

# Adding white gaussian noise
def awgn(x, snr):
  signal_power = np.mean(x ** 2)
  noise_power = signal_power / (10 ** (snr / 10.0))
  noise = np.random.normal(scale=np.sqrt(noise_power), size=x.shape)
  x_with_noise = x + noise
  return x_with_noise

# FISTA reconstruction functions
def gradientImage(I):
  Ix = np.zeros(I.shape)
  Iy = np.zeros(I.shape)
  Ix[:, :-1] = I[:, 1:] - I[:, :-1]
  Iy[:-1, :] = I[1:, :] - I[:-1, :]
  return (Ix, Iy)

def divergenceImage(vx, vy):
  dv = np.zeros(vx.shape)
  dv[:, 1:] = dv[:, 1:] + vx[:, 1:] - vx[:, :-1]
  dv[1:, :] = dv[1:, :] + vy[1:, :] - vy[:-1, :]
  return dv

def totalVariationImage(I):
  (Ix, Iy) = gradientImage(I)
  TV = np.sum(np.sqrt(Ix**2 + Iy**2), axis=None)
  return TV

def proximalTV(I, Lambda, Iter):
# We find vx and vy by minimizing ||I - lambda div(vx, vy)||2 under the
# condition of ||(vx, vy)|| <= 1
# We can minimize this by gradient descent
# Then the new image is just I - lambda div(vx, vy) and its total
# variation can be proved to decrease after this step because this new I is
# the optimization point of ||I_new - I||2 + 2 lambda TV(I_new)
#
# For more details about the algorithm, please refer to this paper: Beck,
# Amir, and Marc Teboulle. "Fast gradient-based algorithms for constrained
# total variation image denoising and deblurring problems." IEEE
# transactions on image processing 18.11 (2009): 2419-2434.
  lr = 1/(16 * Lambda)
  L = np.zeros(Iter)
  (I_x, I_y) = gradientImage(I)
  vx = np.zeros(I.shape)
  vy = np.zeros(I.shape)
  # For the acceleration step
  acvx = np.zeros(I.shape)
  acvy = np.zeros(I.shape)

  T = 1
  for k in range(0, Iter):
    T_pre = T
    T = (1 + np.sqrt(1 + 4 * T**2)) / 2
    vx_pre = vx
    vy_pre = vy

    dv = divergenceImage(acvx, acvy)
    # The derivative for gradient descent
    I_p = I - Lambda * dv
    I_p[I_p < 0] = 0 # Nonnegativity constraint
    (G_x, G_y) = gradientImage(I_p)
    G_x = 2 * G_x
    G_y = 2 * G_y

    # L is the thing we want to minimize
    L[k] = np.sum(I_p ** 2, axis=None)

    # Gradient descent
    vx = acvx - lr * G_x
    vy = acvy - lr * G_y

    vm = np.sqrt(vx**2 + vy**2)
    pr = vm;
    pr[vm <= 1] = 1
    vx = vx / pr
    vy = vy / pr

    # FISTA acceleration
    acvx = (T_pre - 1) / T * (vx - vx_pre) + vx
    acvy = (T_pre - 1) / T * (vy - vy_pre) + vy

  dv = divergenceImage(vx, vy)
  I_p = I - Lambda * dv
  I_p[I_p < 0] = 0 # Nonnegativity constraint


  TV = totalVariationImage(I)
  TV_p = totalVariationImage(I_p)
  Dis = np.sum((I_p - I) ** 2, axis=None)
  print("----Dual problem objective: before is %f" %(L[0]))
  print("----Dual problem objective: middle is %f" %(L[round(Iter/2)]))
  print("----Dual problem objective: after is %f" %(L[-1]))
  print("----Before proximal, ||I - I||2 + 2 lambda TV(I) is %f" %(2 * Lambda * TV))
  print("----After proximal, ||I_p - I||2 + 2 lambda TV(I_p) is %f" %(Dis + 2 * Lambda * TV_p))

  return I_p

def reconImageFISTA(A, b, M, N, Lambda, L, Iter, Iter_sub):

  T = 1
  Loss = np.zeros(Iter)
  x = np.zeros(M * N)
  y = np.zeros(M * N)
  for k in range(0, Iter):
    print("==========FISTA iteration number %d / %d==========" %(k, Iter))
    T_pre = T
    x_pre = x
    # Compute x = x - 2 / L * AT(Ax - b)
    Res = np.sum((np.dot(A, y) - b) ** 2, axis=None)
    print("Before gradient step the Residule is %f" %(Res))
    y = y - (2 / L) * np.transpose(A) @ (A @ y - b)
    Res = np.sum((np.dot(A, y) - b) ** 2, axis=None)
    print("After gradient step the Residule is %f" %(Res))

    # Compute x = TVdenoise(x)
    I = y.reshape(N, M)
    TV = totalVariationImage(I)
    print("Before TV denoising TV is %f" %(TV))
    I = proximalTV(I, Lambda * 2 / L, Iter_sub)
    TV = totalVariationImage(I)
    print("After TV denoising TV is %f" %(TV))
    x = I.reshape(N * M)

    # Do FISTA acceleration
    T = (1 + np.sqrt(1 + 4 * T**2)) / 2
    y = x + (T_pre - 1) / T * (x - x_pre)

    I = x.reshape(N, M)
    Res = np.sum((np.dot(A, x) - b) ** 2, axis=None)
    TV = totalVariationImage(I)
    Loss[k] = Res + 2 * Lambda * TV
    print("The loss function is %f, Residule is %f, TV is %f" %(Loss[k], Res, TV))
  return (I, Loss)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Reconstruct")
   

    parser.add_argument('--r-ring', type=float, default=0.11, help='')
    

    parser.add_argument('--experiment-name', type=str, help='Name of experiment. This will be used to create a folder in results/ and checkpoints/')
    parser.add_argument('--resume-exp', action="store_true", help='Name of experiment. This will be used to create a folder in results/ and checkpoints/')

    args = parser.parse_args()
    
    # Fixed parameters
    R_ring      = 0.11
    M         = 100
    N         = 100
    V_sound      = 1500
    px        = 1e-3
    py        = 1e-3

    # Free parameters
    N_transducer   = 100   #512 in real experimental setting
    dt        = 0.5e-6  #0.025e-6 in real experimental setting

    center_freq   = 0.5e6 #2.25e6 in real experimental setting
    fwhm       = 0.5e6 #2.16e6 in real experimental setting

    oversample    = 10
    padsample    = 50

    with open('/central/groups/mlprojects/pat/pat_norm_crop/config.json') as f:
        cfg= json.load(f)
        print("config.json loaded!")
    imgSize = 64

    Diag = np.sqrt((py * (M / 2 - 0.5))**2 + (px * (N / 2 - 0.5))**2)
    T_min = (R_ring - Diag) / V_sound
    T_max = (R_ring + Diag) / V_sound

    T_max_n = round(T_max / dt)
    T_min_n = round(T_min / dt)
    T_min  = T_min_n * dt
    N_sample = T_max_n - T_min_n + 1

    print("Start computing A")
    if os.path.exists("./A.pickle"):
        with open('./A.pickle', 'rb') as file:
            As = pickle.load(file)
        A, x_sample, y_sample, t_sample = As[0], As[1], As[2], As[3]
    else:
        (A, x_sample, y_sample, t_sample) = forwardMatrixFullRingCDMMI(N_transducer, R_ring, px, py, M, N, dt, N_sample, oversample, padsample, T_min, center_freq, fwhm, V_sound)
        with open('./A.pickle', 'wb') as file:
            pickle.dump([A, x_sample, y_sample, t_sample], file)

    print("Finished computing A")

    #plt.figure(figsize = (30,30))
    #plt.imshow(A.T)

    # Compute the norm of the norm of the matrix for later use.
    # Be aware that this may take very long time! But it only needs to be computed once
    Anorm = np.linalg.norm(A, ord = 2)
    print(f"Anorm = {Anorm}")

    print("Load Image")
    data = scipy.io.loadmat('/central/groups/mlprojects/pat/PAT_dataset/2/img.mat')
    I_gt = data['img']

    x_gt = I_gt.reshape(M * N)
    b = np.transpose(A)@x_gt
    # b = awgn(b, 10)

    # b = b.reshape((N_transducer, t_sample.shape[0]))
    # b = b[cfg["left"]:cfg["left"]+imgSize, cfg["top"]: cfg["top"]+imgSize]
    # b = b.reshape((imgSize * imgSize))
    # A = A.reshape((M, N, N_transducer, t_sample.shape[0]))
    # A = A[:,:, cfg["left"]:cfg["left"]+imgSize, cfg["top"]: cfg["top"]+imgSize]
    # A = A.reshape((M * N, imgSize * imgSize))

    # This is the ground truth of the image
    # plt.figure()
    # plt.imshow(I_gt)

    # S = b.reshape(N_transducer, t_sample.shape[0]).T;

    # S_sa = np.zeros((t_sample.shape[0], N_transducer));
    # S_sa[:, ::4] = S[:, ::4]

    # S_lv = np.zeros((t_sample.shape[0], N_transducer));
    # S_lv[:, :50] = S[:, :50]

    # # This is the signal
    # plt.figure()
    # plt.imshow(S_sa)
    # plt.axis('off')

    # plt.figure()
    # plt.imshow(S_lv)
    # plt.axis('off')

    measurements = np.load('/central/groups/mlprojects/pat/pat_norm_crop/test/signal_1.npy')
    globalMax = cfg['max']
    globalMin = cfg['min']
    measurements = (measurements + 1) / 2 * (globalMax - globalMin) + globalMin

    Iter = 100
    Iter_sub = 1000
    Lambda = 0.001
    Anorm = 0.02
    L = 2 * Anorm * Anorm

    (I, Loss) = reconImageFISTA(np.transpose(A), b, M, N, Lambda, L, Iter, Iter_sub)

    fig = plt.figure()
    plt.imshow(I_gt)
    fig.savefig("I_gt.png")
    fig2 = plt.figure()
    plt.imshow(I)
    fig2.savefig("I.png")