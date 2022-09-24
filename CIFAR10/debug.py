# import torch
# import numpy as np


# def proj_simplex(v, epsilon=12):
#     assert epsilon > 0, "Radius s must be strictly positive (%d <= 0)" % s
#     batch_size = v.shape[0]
#     # check if we are already on the simplex    
#     '''
#     #Not checking this as we are calling this from the previous function only
#     if v.sum(dim = (1,2,3)) == s and np.alltrue(v >= 0):
#         # best projection: itself!
#         return v
#     '''
#     # get the array of cumulative sums of a sorted (decreasing) copy of v
#     gamma = v.view(batch_size,-1)
#     n = gamma.shape[1]
#     # print(u)
#     # print(u.shape)
#     # input('check')
#     gamma, indices = torch.sort(gamma, descending = True)
#     # print('gamma', gamma)
#     gamma_cumsum = gamma.cumsum(dim = 1)
#     js = 1.0 / torch.arange(1, n+1).float()
#     temp = gamma - js * (gamma_cumsum - epsilon)
#     rho = (torch.argmin((temp > 0).int().detach().cpu(), dim=1) - 1 ) % n
#     print('temp', (temp > 0).int().detach().cpu())
#     print('rho', rho)
#     # print('check1')
#     # print(gamma_cumsum)
#     # print(rho.shape)
#     # print(gamma_cumsum[rho.unsqueeze(0)])
#     # print(gamma_cumsum[rho])
#     rho_index = torch.stack([torch.arange(batch_size), rho], dim=0).numpy()
#     print(rho_index)
#     # print('rho_index', rho_index)
#     # print(gamma_cumsum[rho_index])
#     # print(gamma_cumsum[1, 1])
#     # print(rho.float().shape)
#     eta = (1.0 / (1 + rho.float()) * (gamma_cumsum[rho_index] - epsilon)).unsqueeze(1)
#     new_delta = torch.clamp(gamma - eta, 0)
#     # print('check')
#     # print(eta)
#     # print(gamma.shape)
#     # print(eta.shape)
#     # print(gamma - eta)

#     # u = comp.cumsum(dim = 2)
#     # w = (comp-1).cumsum(dim = 2)
#     # print(u)
#     # print(w)
#     # u = u + w
#     # rho = torch.argmax(u, dim = 2)
#     # print(u)
#     # print(rho)
#     # rho = rho.view(batch_size)
#     # c = torch.FloatTensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ])
#     # c = c-s
#     # # compute the Lagrange multiplier associated to the simplex constraint
#     # theta = torch.div(c,(rho.float() + 1))
#     # theta = theta.view(batch_size,1,1,1)
#     # # compute the projection by thresholding v using theta
#     # w = (v - theta).clamp(min=0)
#     return new_delta

# def proj_l1(x, eps_d=2):
#     n = x.shape[0]
#     d = x.reshape(n, -1)  # n * N (N=h*w*ch)
#     abs_d = np.abs(d)  # n * N
#     mu = -np.sort(-abs_d, axis=-1)  # n * N
#     cumsums = mu.cumsum(axis=-1)  # n * N
#     js = 1.0 / np.arange(1, 4 + 1)
#     temp = mu - js * (cumsums - eps_d)
#     print(temp)
#     print(np.argmin(temp > 0, axis=-1))
#     rho = torch.argmax((temp > 0).int().detach().cpu(), dim=1)
#     print(rho)
#     print(cumsums[range(n), rho])
#     theta = 1.0 / (1 + rho) * (cumsums[range(n), rho] - eps_d)
#     sgn = np.sign(d)
#     d = sgn * np.maximum(abs_d - np.expand_dims(theta, -1), 0)
#     x = d.reshape(-1, 4)

#     return x

# a = torch.tensor([[4, 3, 2.5, 1],
# [5, 4, 2.5, 1],
# ])

# a = torch.tensor([[
#         0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1360, 0.1250, 0.1250, 0.1250,
#          0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1193, 0.1167, 0.1167, 0.1167,
#          0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111,
#          0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
#          0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
#          0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
#          0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
#          0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
#          0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
#          0.0833, 0.0769, 0.0769, 0.0769, 0.0769, 0.0769, 0.0769, 0.0769, 0.0769,
#          0.0769, 0.0769, 0.0769, 0.0769, 0.0769, 0.0714, 0.0714, 0.0714, 0.0714,
#          0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714,
#          0.0714, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667,
#          0.0667, 0.0667, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526,
#          0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526,
#          0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500,
#          0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0000, 0.0000,],
#         [0.1714, 0.1667, 0.1667, 0.1667, 0.1611, 0.1484, 0.1333, 0.1333, 0.1250,
#          0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1216, 0.1167, 0.1111, 0.1111,
#          0.1111, 0.1111, 0.1111, 0.1111, 0.1098, 0.1020, 0.1000, 0.1000, 0.1000,
#          0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.0980, 0.0863, 0.0833, 0.0833,
#          0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
#          0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
#          0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
#          0.0833, 0.0833, 0.0824, 0.0769, 0.0769, 0.0769, 0.0769, 0.0769, 0.0769,
#          0.0745, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0706,
#          0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667,
#          0.0667, 0.0611, 0.0588, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526,
#          0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526,
#          0.0526, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500,
#          0.0500, 0.0500, 0.0471, 0.0471, 0.0431, 0.0392, 0.0392, 0.0314, 0.0314,
#          0.0275, 0.0275, 0.0275, 0.0275, 0.0275, 0.0275, 0.0243, 0.0235, 0.0196,
#          0.0196, 0.0167, 0.0157, 0.0157, 0.0157, 0.0118, 0.0118, 0.0118, 0.0118,
#          0.0078, 0.0078, 0.0064, 0.0039, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,]])

# proj = proj_simplex(a, 12)

# # print(proj_simplex(a, 12))

# print(torch.sum(proj, dim=1))

# # a = torch.tensor([[5, 3, 3, 0.5]]).numpy()

# # print(proj_l1(a, 2))


# ---------------------------------
import numpy as np

e1 = 2.
einf = 2 / 255.

alpha = e1 / einf - int(e1 / einf)

e2 = e1 / np.sqrt(e1 / einf - alpha + alpha**2)

print(alpha, e2)