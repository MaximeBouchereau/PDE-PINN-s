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
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point

from itertools import product
import statistics

import sys
import time
import datetime
from datetime import datetime as dtime

# Python code to solve 1D Schrödinger equation with PINN's for Dirichlet boundary conditions
print(150 * "_")
print(" ")
print("   CODE FOR SCHRÖDINGER'S EQUATION CONTROLABILITY")
print(150 * "_")


L = np.pi    # Size of the (space) domain of resolution

params_0 = {'name': 'proper_mode_0', 'initial_condition': [1, 0], 'final_condition': [1, 0.1]}
params_1 = {'name': 'proper_mode_1', 'initial_condition': [0, 1], 'final_condition': [0.1, 1]}
params = params_0

class NN(nn.Module):
    """Class of Neural Networks and main functions used in this scipt"""

    def __init__(self):
        zeta , HL = 256 , 4
        super().__init__()
        self.PSI = nn.ModuleList([nn.Linear(2, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, 2, bias=True)])
        self.U = nn.ModuleList([nn.Linear(1, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, 1, bias=True)])

    def f_0(self, x):
        """First proper mode of quantum harmonic oscillator (hamiltonian).
        Inputs:
        - x: Tensor of shape (1,n) - Space variable (n is the batch size for vectorized computation)."""
        return (1 / torch.pi) ** 0.25 * torch.exp(-0.5 * x ** 2)

    def f_1(self, x):
        """First proper mode of quantum harmonic oscillator (hamiltonian).
        Inputs:
        - x: Tensor of shape (1,n) - Space variable (n is the batch size for vectorized computation)."""
        return (torch.sqrt(torch.tensor(2)) /torch.pi ** 0.25) * x * torch.exp(-0.5 * x ** 2)

    def psi_0(self, x):
        """Initial condition function.
        Inputs:
        - x: Tensor of shape (1,n) - Space variable (n is the batch size for vectorized computation)."""
        alpha, beta = params['initial_condition']
        norm = torch.norm(torch.tensor([alpha, beta]).float())
        alpha, beta = alpha / norm, beta / norm
        return alpha * self.f_0(x) + beta * self.f_1(x)

    def psi_1(self, x):
        """Final condition function.
        Inputs:
        - x: Tensor of shape (1,n) - Space variable (n is the batch size for vectorized computation)."""
        alpha, beta = params['final_condition']
        norm = torch.norm(torch.tensor([alpha, beta]).float())
        alpha, beta = alpha / norm, beta / norm
        return alpha * self.f_0(x) + beta * self.f_1(x)

    def forward(self, t, x):
        """Structured Neural Network.
        Inputs:
         - t: Tensor of shape (1,n) - time variable
         - x: Tensor of shape (1,n) - space variable
         """

        t , x = t.float().T , x.float().T

        psi, u = torch.cat((t,x) , dim=1), t

        # Structure of the solution/control of the PDE
        for i, module in enumerate(self.PSI):
            psi = module(psi)

        for i, module in enumerate(self.U):
            u = module(u)

        psi, u = psi.T, u.T

        return psi, u

class ML(NN):
    """Training of the neural network for solving 1D heat equation"""

    def Loss_autograd(self, t, x, T, model):
        """Computes the Loss function associated with the PINN
        Inputs:
        - t: Tensor of shape (1,n): Inputs of Neural Network - time variable (n is the size of the batch)
        - x: Tensor of shape (1,n): Inputs of Neural Network - space variable (n is the size of the batch)
        - T: Float - Final time of PDE control
        - model: Neural network which will be optimized
        Computes a predicted value uhat which is a tensor of shape (1,n) and returns the mean squared error between Yhat and Y.
        Derivatives are approximated with autograd.
        => Returns a tensor of shape (1,1)"""

        t = torch.tensor(t, dtype=torch.float32)
        t.requires_grad = True

        x = torch.tensor(x, dtype=torch.float32)
        x.requires_grad = True

        ones = torch.ones_like(x)

        x_dom = torch.linspace(-L, L, 100).unsqueeze(0)
        ones_dom = torch.ones_like(x_dom)

        u_hat = torch.zeros_like(x)
        u_hat.requires_grad = True

        psi_hat, u_hat = model(t, x)

        psi1_hat = psi_hat[0,:]
        psi2_hat = psi_hat[1,:]

        psi1_hat_x = torch.autograd.grad(psi1_hat, x, grad_outputs=torch.ones_like(psi1_hat), create_graph=True)[0]
        psi1_hat_xx = torch.autograd.grad(psi1_hat_x, x, grad_outputs=torch.ones_like(psi1_hat_x), create_graph=True)[0]

        psi2_hat_x = torch.autograd.grad(psi2_hat, x, grad_outputs=torch.ones_like(psi2_hat), create_graph=True)[0]
        psi2_hat_xx = torch.autograd.grad(psi2_hat_x, x, grad_outputs=torch.ones_like(psi2_hat_x), create_graph=True)[0]

        psi1_hat_t = torch.autograd.grad(psi1_hat, t, grad_outputs=torch.ones_like(psi1_hat), create_graph=True)[0]
        psi2_hat_t = torch.autograd.grad(psi2_hat, t, grad_outputs=torch.ones_like(psi2_hat), create_graph=True)[0]

        loss_PDE = ((psi1_hat_t + 0.5 * psi2_hat_xx - 0.5 * x ** 2 * psi2_hat - u_hat * x * psi2_hat) ** 2 + (- psi2_hat_t + 0.5 * psi1_hat_xx - 0.5 * x ** 2 * psi1_hat - u_hat * x * psi1_hat) ** 2 ).mean() # Loss associated to the PDE

        psi_hat_L, psi_hat_R = model(t, -L*ones)[0], model(t, L*ones)[0]
        loss_BC = ((psi_hat_L).abs() ** 2).mean() + ((psi_hat_R).abs() ** 2).mean() # Loss associated to boundary conditions (Dirichlet)

        psi_hat_0 = model(0 * ones, x)[0]
        psi1_hat_0 = psi_hat_0[0,:]
        psi2_hat_0 = psi_hat_0[1,:]
        psi_0 = self.psi_0(x)
        psi_hat_1 = model(T * ones, x)[0]
        psi1_hat_1 = psi_hat_1[0, :]
        psi2_hat_1 = psi_hat_1[1, :]
        psi_1 = self.psi_1(x)

        loss_IC = (((psi1_hat_0 - psi_0)).abs() ** 2).mean() +  ((psi2_hat_0).abs() ** 2).mean() # Loss associated to initial condition
        loss_FC = (((psi1_hat_1 - psi_1)).abs() ** 2).mean() +  ((psi2_hat_1).abs() ** 2).mean() # Loss associated to final condition

        # loss_L2 = sum([(2 * L * (model(t_n * ones_dom, x_dom) ** 2).mean() - 1) ** 2 for t_n in torch.linspace(0, t.max(), 10)]) # Constant norm
        # loss_L2 = 0

        # print(2 * L * (model(ones_dom, x_dom) ** 2).mean())

        # print("Loss_PDE:", format(loss_PDE, '.4E'), " - Loss_BC:", format(loss_BC, '.4E'), " - Loss_IC:", format(loss_IC, '.4E'))
        # print("Loss_PDE:", format(loss_PDE, '.4E'), " - Loss_BC:", format(loss_BC, '.4E'), " - Loss_IC:", format(loss_IC, '.4E'), " - Loss_L2:", format(loss_L2, '.4E'))

        return loss_PDE + loss_BC + loss_IC + loss_FC, loss_PDE, loss_BC, loss_IC, loss_FC

    def Train_autograd(self, model , T = 1 , K = 1000 , BS = 64 ,  N_epochs = 100 , N_epochs_print = 10):
        """Makes the training of the model to learn solution of PDE
        Inputs:
        - model: Neural network which will be optimized
        - T: Float - Length of the time interval. Default: 1
        - K: Int - Number of data for both train and test. Default: 1000
        - BS: Int - Batch size. Default: 64
        - N_epochs: Int - Number of epochs for gradient descent. Default: 100
        - N_epochs_print: Int - Number of epochs between two prints of the Loss. Default: 10
        Derivatives are approximated with autograd.
        => Returns the lists Loss_train and Loss_test of the values of the Loss w.r.t. training and test,
        and best_model, which is the best apporoximation of the desired model"""

        start_time_train = time.time()

        print(" ")
        print(150 * "-")
        print("Training...")
        print(150 * "-")

        #t_train , t_test = torch.linspace(0,T,K).reshape([1,K]) , T*torch.rand([1 , K])
        #x_train , x_test = torch.linspace(-1,1,K).reshape([1,K]) , 2*torch.rand([1 , K])-1

        t_train, t_test = T * torch.rand([1, K]), T * torch.rand([1, K])
        # t_train, t_test = torch.linspace(0, T, K).unsqueeze(0), T * torch.rand([1, K])
        x_train, x_test = 2 * L * torch.rand([1, K]) - L, 2 * L * torch.rand([1, K]) - L
        # x_train, x_test = torch.linspace(-L, L, K).unsqueeze(0), torch.sort(2 * L * torch.rand([1, K]) - L)[0]

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-9, amsgrad=True)  # Algorithm AdamW
        best_model, best_loss_train, best_loss_test = model, np.infty, np.infty  # Selects the best minimizer of the Loss function
        Loss_train, Loss_train_PDE, Loss_train_BC, Loss_train_IC, Loss_train_FC = [], [], [], [], [] # list for loss_train values
        Loss_test, Loss_test_PDE, Loss_test_BC, Loss_test_IC, Loss_test_FC = [], [], [], [], [] # list for loss_test values

        for epoch in range(N_epochs + 1):
            for ixs in torch.split(torch.arange(t_train.shape[1]), BS):
                optimizer.zero_grad()
                model.train()
                t_batch = t_train[:, ixs]
                x_batch = x_train[:, ixs]
                loss_train = self.Loss_autograd(t_batch, x_batch, T,model)[0]
                loss_train.backward()
                optimizer.step()  # Optimizer passes to the next epoch for gradient descent

            loss_train, loss_train_PDE, loss_train_BC, loss_train_IC, loss_train_FC = self.Loss_autograd(t_train, x_train, T, model)
            loss_test, loss_test_PDE, loss_test_BC, loss_test_IC, loss_test_FC = self.Loss_autograd(t_test, x_test, T, model)

            if loss_train < best_loss_train:
                best_loss_train = loss_train
                best_loss_test = loss_test
                best_model = copy.deepcopy(model)
                # best_model = model

            Loss_train.append(loss_train.item())
            Loss_train_PDE.append(loss_train_PDE.item())
            Loss_train_BC.append(loss_train_BC.item())
            Loss_train_IC.append(loss_train_IC.item())
            Loss_train_FC.append(loss_train_FC.item())
            Loss_test.append(loss_test.item())
            Loss_test_PDE.append(loss_test_PDE.item())
            Loss_test_BC.append(loss_test_BC.item())
            Loss_test_IC.append(loss_test_IC.item())
            Loss_test_FC.append(loss_test_FC.item())

            if epoch % N_epochs_print == 0:  # Print of Loss values (one print each N_epochs_print epochs)
                end_time_train = start_time_train + ((N_epochs + 1) / (epoch + 1)) * (time.time() - start_time_train)
                end_time_train = datetime.datetime.fromtimestamp(int(end_time_train)).strftime(' %Y-%m-%d %H:%M:%S')
                print('    Epoch', epoch, ': Loss_train =', format(loss_train, '.4E'), ': Loss_test =', format(loss_test, '.4E'), " -  Estimated end:", end_time_train)
                print('       > Train: PDE:', format(loss_train_PDE, '.2E'), ' - BC:', format(loss_train_BC, '.2E'), ' - IC:', format(loss_train_IC, '.2E'), ' - FC:', format(loss_train_FC, '.2E'))
                print('       > Test: PDE:', format(loss_test_PDE, '.2E'), ' - BC:', format(loss_test_BC, '.2E'), ' - IC:', format(loss_test_IC, '.2E'), ' - FC:', format(loss_test_FC, '.2E'))

        print("Loss_train (final)=", format(best_loss_train, '.4E'))
        print("Loss_test (final)=", format(best_loss_test, '.4E'))

        print("Computation time for training (h:min:s):", str(datetime.timedelta(seconds=int(time.time() - start_time_train))))

        torch.save((Loss_train, Loss_train_PDE, Loss_train_BC, Loss_train_IC, Loss_train_FC, Loss_test, Loss_test_PDE, Loss_test_BC, Loss_test_IC, Loss_test_FC, best_model , T) , "model_Control_Schrodinger_Equation_1D")

        pass

    def Integrate(self, name_model="model_Control_Schrodinger_Equation_1D",  ht = 0.02 ,  hx = L/50, save_fig=False):
        """Integrates the PDE with trained model.
        Inputs:
        - name_model: Str - Name of the trained model. Default: model_Heat_Equation_1D
        - ht: Float - Step size for time. Default: 0.02
        - hx: Float - Step size for space. Default: 0.02
        - save_fig: Boolean - Saves the figure or not. Default: False"""

        Loss_train, Loss_train_PDE, Loss_train_BC, Loss_train_IC, Loss_train_FC, Loss_test, Loss_test_PDE, Loss_test_BC, Loss_test_IC, Loss_test_FC, model, T = torch.load(name_model)

        t_grid, x_grid = torch.arange(0, T+ht, ht), torch.arange(-L, L+hx, hx)
        grid_t, grid_x = torch.meshgrid(t_grid, x_grid)

        # Resolution with Crank-Nicolson scheme
        print("   > Resolution with Crank-Nicolson scheme...")
        N , J = t_grid.shape[0] , x_grid.shape[0]
        A0 = (1j)*(0.5*ht/hx**2)*(2*torch.diag(torch.ones(J-2),0) - torch.diag(torch.ones(J-3),-1) - torch.diag(torch.ones(J-3),1)) + 0.5*(1j)*ht*torch.diag(x_grid[1:-1]**2)
        z_CN = torch.zeros_like(grid_t, dtype=torch.cfloat)
        ones = torch.ones_like(x_grid)

        z_CN[0,:] = self.psi_0(x_grid)
        for n in range(N-1):
            count = int(100 * ((n+2) / N))
            sys.stdout.write("\r%d " % count + "%")
            sys.stdout.flush()
            u_hat = model(t_grid[n] * ones[1:-1].unsqueeze(0), x_grid[1:-1].unsqueeze(0))[1]
            A = A0 + (1j) * ht * torch.diag((u_hat * x_grid[1:-1]).squeeze())
            z_CN[n+1,1:J-1] = torch.inverse(torch.eye(J-2)+0.5*A)@(torch.eye(J-2)-0.5*A)@z_CN[n,1:J-1]

        z_CN_ = (z_CN.real)**2 + (z_CN.imag)**2
        z_CN_Re = z_CN.real
        z_CN_Im = z_CN.imag

        z_CN_ = z_CN_.detach().numpy()
        z_CN_Re = z_CN_Re.detach().numpy()
        z_CN_Im = z_CN_Im.detach().numpy()

        # Resolution with the PINN
        print(" ")
        print("Resolution with PINN...")
        z_PINN = torch.zeros_like(grid_t)
        z_PINN_Re = torch.zeros_like(grid_t)
        z_PINN_Im = torch.zeros_like(grid_t)
        ones = torch.ones_like(x_grid).unsqueeze(0)
        for it in range(grid_t.shape[0]):
            count = int(100 * ((it+1) / torch.numel(t_grid)))
            sys.stdout.write("\r%d " % count + "%")
            sys.stdout.flush()
            psi = model(t_grid[it] * ones, x_grid.unsqueeze(0))[0]
            z_PINN[it, :] = psi[0, :] ** 2 + psi[1, :] ** 2
            z_PINN_Re[it, :] = psi[0, :]
            z_PINN_Im[it, :] = psi[1, :]
            # z_PINN[it, :] = psi[0, :]
            # for ix in range(grid_x.shape[1]):
            #     z_PINN[it, ix] = model(torch.tensor([[grid_t[it, ix]]]), torch.tensor([[grid_x[it, ix]]]))[0]**2 + model(torch.tensor([[grid_t[it, ix]]]), torch.tensor([[grid_x[it, ix]]]))[1]**2

        z_PINN = z_PINN.detach().numpy()
        z_PINN_Re = z_PINN_Re.detach().numpy()
        z_PINN_Im = z_PINN_Im.detach().numpy()


        # Distance between both solutions
        z_DIFF = np.abs(z_CN_-z_PINN)

        plt.figure(figsize=(25,12))

        plt.subplot(2, 4, 1)
        plt.imshow(z_PINN.T,cmap="jet" , aspect="auto" , extent=(0,T,-L,L))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("$|\psi|^2$ (PINN)")

        plt.subplot(2, 4, 2)
        plt.imshow(z_CN_.T, cmap="jet", aspect="auto", extent=(0, T, -L, L))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("$|\psi|^2$ (Crank-Nicolson)")

        plt.subplot(2, 4, 3)
        plt.imshow(z_DIFF.T, cmap="spring", aspect="auto", extent=(0, T, -L, L))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("Difference")

        plt.subplot(2, 4, 4)
        plt.plot(list(range(len(Loss_train))), Loss_train, label="$Loss_{train}$", color="green")
        plt.plot(list(range(len(Loss_test))), Loss_test, label="$Loss_{test}$", color="red")
        plt.yscale("log")
        plt.xlabel("epochs")
        # plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss evolution")
        plt.grid()

        plt.subplot(2, 4, 5)
        plt.imshow(z_PINN_Re.T, cmap="jet", aspect="auto", extent=(0, T, -L, L))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("$Re(\psi)$ (PINN)")

        plt.subplot(2, 4, 6)
        plt.imshow(z_CN_Re.T, cmap="jet", aspect="auto", extent=(0, T, -L, L))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("$Re(\psi)$ (Crank-Nicolson)")

        plt.subplot(2, 4, 7)
        plt.imshow(z_PINN_Im.T, cmap="jet", aspect="auto", extent=(0, T, -L, L))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("$Im(\psi)$ (PINN)")

        plt.subplot(2, 4, 8)
        plt.imshow(z_CN_Im.T, cmap="jet", aspect="auto", extent=(0, T, -L, L))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("$Im(\psi)$ (Crank-Nicolson)")

        if save_fig == False:
            plt.show()
        else:
            plt.savefig("PINN_Control_Schrodinger_Equation_1D.pdf" , dpi = (500))
        pass