import warnings

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

import torch
import torch.optim as optim
import torch.nn as nn
import copy

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point

from itertools import product
import statistics

import sys
import time
import datetime
from datetime import datetime as dtime

# Python code to solve 2D Schrödinger equation with PINN's

L = 2    # Size of the (space) domain of resolution

class NN(nn.Module):
    """Class of Neural Networks used in this scipt"""

    def u0(self, x, y):
        """Initial (real valued) condition function.
        Inputs:
        - x: Tensor of shape (1,n) - space variable, first dimension
        - y: Tensor of shape (1,n) - space variable, second dimension
        """
        ones = torch.ones_like(x)
        #return torch.exp(-10*(x**2+y**2))
        return 0.5*torch.exp(-10*((x-L*ones/3)**2+(y-L*ones/3)**2))

    def __init__(self):
        zeta , HL = 128 , 2
        super().__init__()
        self.U1 = nn.ModuleList([nn.Linear(3, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, 1, bias=True)])
        self.U2 = nn.ModuleList([nn.Linear(3, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, 1, bias=True)])

    def forward(self, t, x, y):
        """Structured Neural Network.
        Inputs:
         - t: Tensor of shape (1,n) - time variable
         - x: Tensor of shape (1,n) - space variable, first dimension
         - y: Tensor of shape (1,n) - space variable, second dimension
         """

        t, x, y = t.float().T, x.float().T, y.float().T

        X1 = torch.cat((t,x,y) , dim=1)
        X2 = torch.cat((t,x,y) , dim=1)

        # Structure of the solution of the equation

        for i, module in enumerate(self.U1):
            X1 = module(X1)
        for i, module in enumerate(self.U2):
            X2 = module(X2)

        return X1.T, X2.T

class ML(NN):
    """Training of the neural network for solving 1D heat equation"""


    def Loss(self, t, x, y, model):
        """Computes the Loss function associated with the PINN
        Inputs:
        - t: Tensor of shape (1,n): Inputs of Neural Network - time variable
        - x: Tensor of shape (1,n): Inputs of Neural Network - space variable, first dimension
        - y: Tensor of shape (1,n): Inputs of Neural Network - space variable, second dimension
        - model: Neural network which will be optimized
        Computes a predicted value uhat which is a tensor of shape (1,n) and returns the mean squared error between Yhat and Y
        => Returns a tensor of shape (1,1)"""

        t = torch.tensor(t, dtype=torch.float32)
        t.requires_grad = True

        x = torch.tensor(x, dtype=torch.float32)
        x.requires_grad = True

        y = torch.tensor(y, dtype=torch.float32)
        y.requires_grad = True

        ones = torch.ones_like(x)

        u_hat = torch.zeros_like(x)
        u_hat.requires_grad = True

        delta_t = 1e-3  # Small parameter for finite differences in time
        delta_x = 1e-3 # Small parameter for finite differences in space, first dimension
        delta_y = 1e-3 # Small parameter for finite differences in space, second dimension

        u1_hat = model(t, x, y)[0]
        u1_hat_t = (model(t + delta_t * ones, x, y)[0] - model(t - delta_t * ones, x, y)[0]) / (2 * delta_t)
        u1_hat_xx = (model(t, x + delta_x * ones, y)[0] - 2 * model(t, x, y)[0] + model(t, x - delta_x * ones, y)[0]) / (delta_x ** 2)
        u1_hat_yy = (model(t, x, y + delta_y * ones)[0] - 2 * model(t, x, y)[0] + model(t, x, y - delta_y * ones)[0]) / (delta_y ** 2)

        u2_hat = model(t, x, y)[1]
        u2_hat_t = (model(t + delta_t * ones, x, y)[1] - model(t - delta_t * ones, x, y)[1]) / (2 * delta_t)
        u2_hat_xx = (model(t, x + delta_x * ones, y)[1] - 2 * model(t, x, y)[1] + model(t, x - delta_x * ones, y)[1]) / (delta_x ** 2)
        u2_hat_yy = (model(t, x, y + delta_y * ones)[1] - 2 * model(t, x, y)[1] + model(t, x, y - delta_y * ones)[1]) / (delta_y ** 2)

        loss_PDE_1 = (((-u2_hat_t + (u1_hat_xx + u1_hat_yy) - (x**2+y**2)*u1_hat - 0*(u1_hat**2+u2_hat**2)*u1_hat)).abs() ** 2).mean() # Loss associated to the PDE - Real part
        loss_PDE_2 = (((u1_hat_t + (u2_hat_xx + u2_hat_yy) - (x**2+y**2)*u2_hat - 0*(u1_hat**2+u2_hat**2)*u2_hat)).abs() ** 2).mean() # Loss associated to the PDE - Imaginary part

        #u1_hat_B = model(t, L*ones, y)[0]**2 + model(t, -L*ones, y)[0]**2 + model(t, x, -L*ones)[0]**2 + model(t, x, L*ones)[0]**2
        #u2_hat_B = model(t, L*ones, y)[1]**2 + model(t, -L*ones, y)[1]**2 + model(t, x, -L*ones)[1]**2 + model(t, x, L*ones)[1]**2
        #loss_BC = ((u1_hat_B).abs() + (u2_hat_B).abs()).mean() # Loss associated to boundary conditions (Dirichlet on the unit 2-ball for uniform norm)

        u1_hat_0 = model(0 * ones, x, y)[0]
        u2_hat_0 = model(0 * ones, x, y)[1]
        u_0 = self.u0(x, y)
        loss_IC = (((u1_hat_0 - u_0)).abs() ** 2).mean() + (((u2_hat_0)).abs() ** 2).mean()  # Loss associated to initial condition

        loss_BC = (torch.sum(u_0**2) - torch.sum(u1_hat**2 + u2_hat**2)).abs()

        #print(torch.max((u1_hat_0-u_0).abs()))
        #return loss_PDE_1 + loss_PDE_2 + loss_BC + loss_IC
        #return loss_PDE_1 + loss_PDE_2 + loss_IC
        return loss_IC

    def Train(self, model , T = 1 , K = 1000 , BS = 64 ,  N_epochs = 100 , N_epochs_print = 10):
        """Makes the training of the model to learn solution of PDE
        Inputs:
        - model: Neural network which will be optimized
        - T: Float - Length of the time interval. Default: 1
        - K: Int - Number of data for both train and test. Default: 1000
        - BS: Int - Batch size. Default: 64
        - N_epochs: Int - Number of epochs for gradient descent. Default: 100
        - N_epochs_print: Int - Number of epochs between two prints of the Loss. Default: 10
        => Returns the lists Loss_train and Loss_test of the values of the Loss w.r.t. training and test,
        and best_model, which is the best apporoximation of the desired model"""

        start_time_train = time.time()

        print(" ")
        print(150 * "-")
        print("Training...")
        print(150 * "-")

        t_train, t_test = T * torch.rand([1, K]), T * torch.rand([1, K])
        #r_train, r_test = torch.rand([1, K]), torch.rand([1, K])
        #theta_train, theta_test = 2*np.pi*torch.rand([1, K]), 2*np.pi*torch.rand([1, K])
        #x_train, x_test = r_train*torch.cos(theta_train), r_test*torch.cos(theta_test)
        #y_train, y_test = r_train*torch.sin(theta_train), r_test*torch.sin(theta_test)
        #x_train, x_test = torch.linspace(-L,L,K).resize(1,K), torch.linspace(-L,L,K).resize(1,K)
        x_train, x_test = -L + 2*L*torch.rand([1,K]) , -L + 2*L*torch.rand([1,K])
        y_train, y_test = -L + 2*L*torch.rand([1,K]) , -L + 2*L*torch.rand([1,K])

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-9, amsgrad=True)  # Algorithm AdamW
        best_model, best_loss_train, best_loss_test = model, np.infty, np.infty  # Selects the best minimizer of the Loss function
        Loss_train = [] # list for loss_train values
        Loss_test = []  # List for loss_test values

        for epoch in range(N_epochs + 1):
            for ixs in torch.split(torch.arange(x_train.shape[1]), BS):
                optimizer.zero_grad()
                model.train()
                t_batch = t_train[:, ixs]
                x_batch = x_train[:, ixs]
                y_batch = y_train[:, ixs]
                loss_train = self.Loss(t_batch, x_batch, y_batch, model)
                loss_train.backward()
                optimizer.step()  # Optimizer passes to the next epoch for gradient descent

            loss_test = self.Loss(t_test, x_test, y_test, model)

            if loss_train < best_loss_train:
                best_loss_train = loss_train
                best_loss_test = loss_test
                best_model = copy.deepcopy(model)
                # best_model = model

            Loss_train.append(loss_train.item())
            Loss_test.append(loss_test.item())

            if epoch % N_epochs_print == 0:  # Print of Loss values (one print each N_epochs_print epochs)
                end_time_train = start_time_train + ((N_epochs + 1) / (epoch + 1)) * (time.time() - start_time_train)
                end_time_train = datetime.datetime.fromtimestamp(int(end_time_train)).strftime(' %Y-%m-%d %H:%M:%S')
                print('    Epoch', epoch, ': Loss_train =', format(loss_train, '.4E'), ': Loss_test =', format(loss_test, '.4E'), " -  Estimated end:", end_time_train)

        print("Loss_train (final)=", format(best_loss_train, '.4E'))
        print("Loss_test (final)=", format(best_loss_test, '.4E'))

        print("Computation time for training (h:min:s):", str(datetime.timedelta(seconds=int(time.time() - start_time_train))))

        torch.save((Loss_train, Loss_test, best_model, T) , "model_Schrodinger_Equation_2D")

        pass

    def Integrate(self, name_model="model_Schrodinger_Equation_2D",  ht = 0.1 , hx = L/10 ,  hy = L/10, save_fig=False):
        """Integrates the PDE with trained model.
        Inputs:
        - name_model: Str - Name of the trained model. Default: model_Poisson_Equation_2D
        - ht: Float - Step size for time. Default: 0.1
        - hx: Float - Step size for space variable, first dimension. Default: 0.1
        - hy: Float - Step size for space variable, second dimension. Default: 0.1
        - save_fig: Boolean - Saves the figure or not. Default: False"""

        deltat = 2000*ht

        Loss_train, Loss_test, model, T = torch.load(name_model)

        t_grid, x_grid, y_grid = torch.arange(0, T+ht, ht), torch.arange(-L, L+hx, hx), torch.arange(-L, L+hy, hy)
        grid_x, grid_y = torch.meshgrid(x_grid, y_grid)

        # Resolution with the PINN
        z_PINN = torch.zeros([torch.numel(t_grid), grid_x.size(0), grid_x.size(1)])
        for it in range(torch.numel(t_grid)):
            count = int(100 * (it / torch.numel(t_grid)))
            sys.stdout.write("\r%d " % count + "%")
            sys.stdout.flush()
            for ix in range(grid_x.shape[0]):
                for iy in range(grid_y.shape[1]):
                    z_PINN[it, ix, iy] = model(torch.tensor([[t_grid[it]]]), torch.tensor([[grid_x[ix, iy]]]), torch.tensor([[grid_y[ix, iy]]]))[0]**2 + model(torch.tensor([[t_grid[it]]]), torch.tensor([[grid_x[ix, iy]]]), torch.tensor([[grid_y[ix, iy]]]))[1]**2
                    #z_PINN[it, ix, iy] = model(torch.tensor([[t_grid[it]]]), torch.tensor([[grid_x[ix, iy]]]), torch.tensor([[grid_y[ix, iy]]]))[1]


        z_PINN = z_PINN.detach().numpy()

        fig = plt.figure(figsize=(5,5))
        im = plt.imshow(z_PINN[0, :, :].T, cmap="jet", aspect="equal", extent=(-L, L, -L, L))
        #im = plt.imshow(self.u0(grid_x,grid_y).T, cmap="jet", aspect="equal", extent=(-L, L, -L, L))
        #im = plt.plot(np.cos(theta), np.sin(theta), linestyle="dashed", color="black")
        plt.title("PINN")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.colorbar()


        def animate(n):
            im.set_array(z_PINN[n, :, :].T)
            #im.set_array(torch.tensor(z_PINN[0, :, :].T))
            #im.set_array(torch.tensor(z_PINN[0, :, :].T)-self.u0(grid_x,grid_y).T)
            #im.set_array(self.u0(grid_x,grid_y).T)
            return [im]

        anim = animation.FuncAnimation(fig, animate, frames=torch.numel(t_grid), blit=True, interval=deltat, repeat=True)


        if save_fig == False:
            plt.show()
        else:
            anim.save("PINN_Schrodinger_equation_2D.gif", writer="pillow")
        pass