from typing import Self
from ufl import (TestFunctions, TrialFunction, Identity, as_tensor, as_vector, eq, grad, det, div, dev, inv, tr, sqrt, conditional ,
                gt, dx, inner, derivative, dot, ln, split,acos,cos,sin,lt,
                as_tensor, as_vector, SpatialCoordinate)
from dolfinx.fem import Constant
from petsc4py import PETSc
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, Expression, locate_dofs_topological)


#Class that decribes the thermoelastic axial strain problem
class ThermoElasticAxialSymmetric():


    def __init__(self,u,p,theta,u_old,p_old,theta_old,domain,**kwargs):
        self.k_B      = Constant(domain,1.38E-17)             # Boltzmann's constant
        self.theta0   = Constant(domain,298.0)                  # Initial temperature
        self.Gshear_0 = Constant(domain,280.0)                  # Ground sate shear modulus
        self.N_R      = Constant(domain,PETSc.ScalarType(self.Gshear_0/(self.k_B*self.theta0)))# Number polymer chains per unit ref. volume
        self.lambdaL  = Constant(domain,5.12)                 # Locking stretch
        self.Kbulk    = Constant(domain,PETSc.ScalarType(1000*self.Gshear_0))        # Bulk modulus
        self.alpha    = Constant(domain,180.0E-6)             # Coefficient of thermal expansion
        self.c_v      = Constant(domain,1930.0)                 # Specific heat
        self.k_therm  = Constant(domain,0.16E3)               # Thermal conductivity

        if "alpha" in kwargs.keys():
            self.alpha = kwargs["alpha"]

        self.u=u
        self.p =p
        self.theta= theta

        self.u_old=u_old
        self.p_old =p_old
        self.theta_old= theta_old
        self.x = SpatialCoordinate(domain)
        self.domain = domain
    
    def Kinematics(self, u= None,p=None,theta=None,u_old= None,p_old=None,theta_old=None):
        if u == None:
            u = self.u
        if p == None:
            p = self.p
        if theta == None:
            theta = self.theta
        if u_old == None:
            u_old = self.u_old
        if p_old == None:
            p_old = self.p_old
        if theta_old == None:
            theta_old = self.theta_old
        
        self.F = self.F_ax_calc(u)
        J = det(self.F)   
        #
        self.lambdaBar = self.lambdaBar_calc(u)
        #
        self.F_old = self.F_ax_calc(u_old)
        self.J_old = det(self.F_old)   
        #
        self.C     = self.F.T*self.F
        self.C_old = self.F_old.T*self.F_old

        #  Tmat stress
        Tmat = self.Tmat_calc(u, p, theta)

        # Calculate the stress-temperature tensor
        self.M = self.M_calc(u)

        # Calculate the heat flux
        self.Qmat = self.Heat_flux_calc(u,theta)

        return Tmat,J

    def ax_grad_vector(self,u):
        grad_u = grad(u)
        return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                    [grad_u[1,0], grad_u[1,1], 0],
                    [0, 0, u[0]/self.x[0]]]) 

    # Gradient of scalar field y
    # (just need an extra zero for dimensions to work out)
    def ax_grad_scalar(self,y):
        grad_y = grad(y)
        return as_vector([grad_y[0], grad_y[1], 0.])

    # Axisymmetric deformation gradient 
    def F_ax_calc(self,u):
        dim = len(u)
        Id = Identity(dim)          # Identity tensor
        F = Id + grad(u)            # 2D Deformation gradient
        F33 = 1+(u[0])/self.x[0]      # axisymmetric F33, R/R0    
        F33 = conditional(eq(self.x[0],0),1,F33)
        return as_tensor([[F[0,0], F[0,1],0.0],
                    [F[1,0], F[1,1], 0.0],
                    [0.0,0.0, F33]]) # Full axisymmetric F

    def lambdaBar_calc(self,u):
        F    = self.F_ax_calc(u)
        J    = det(F)
        C    = F.T*F
        Cdis = J**(-2/3)*C
        I1   = tr(Cdis)
        lambdaBar = sqrt(I1/3.0)
        return lambdaBar

    def zeta_calc(self,u):
        lambdaBar = self.lambdaBar_calc(u)
        # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
        z    = lambdaBar/self.lambdaL
        z    = conditional(gt(z,0.95), 0.95, z) # Prevent the function from blowing up
        beta = z*(3.0 - z**2.0)/(1.0 - z**2.0)
        zeta = (self.lambdaL/(3*lambdaBar))*beta
        return zeta


    # Tmat stress 
    def Tmat_calc(self, u, p, theta):
        F = self.F_ax_calc(u)
        C    = F.T*F
        J = det(F)
        #
        zeta = self.zeta_calc(u)
        Gshear  =self.N_R * self.k_B * theta * zeta
        #
        Tmat = J**(-2/3) * Gshear * (F - (1/3)*tr(C)*inv(F.T) ) - J * p * inv(F.T)
        return Tmat

    # Calculate the stress temperature tensor
    def M_calc(self,u):
        Id  = Identity(3)         
        F   = self.F_ax_calc(u) 
        #
        C  = F.T*F
        Cinv = inv(C) 
        J = det(F)
        zeta = self.zeta_calc(u)
        #
        fac1 = self.N_R * self.k_B * zeta
        fac2 = (3*self.Kbulk*self.alpha)/J
        #
        M =  J**(-2/3) * fac1 * (Id - (1/3)*tr(C)*Cinv)  - J * fac2 * Cinv
        return M
        
    #  Heat flux
    def Heat_flux_calc(self,u,theta):
        F = self.F_ax_calc(u) 
        J = det(F)         
        #
        Cinv = inv(F.T*F) 
        #
        Tcond = J * self.k_therm * Cinv # Thermal conductivity tensor
        #
        Qmat = - Tcond * self.ax_grad_scalar(theta)
        return Qmat


    # Calculate  principal Cauchy stresses for visualization only
    def tensor_eigs(self,T):
        # invariants of T
        I1 = tr(T) 
        I2 = (1/2)*(tr(T)**2 - tr(T*T))
        I3 = det(T)
        
        # Intermediate quantities b, c, d
        b = -I1
        c = I2
        d = -I3
        
        # intermediate quantities E, F, G
        E = (3*c - b*b)/3
        F = (2*(b**3) - 9*b*c + 27*d)/27
        G = (F**2)/4 + (E**3)/27
        
        # Intermediate quantities H, I, J, K, L
        H = sqrt(-(E**3)/27)
        I = H**(1/3)
        J = acos(-F/(2*H))
        K = cos(J/3)
        L = sqrt(3)*sin(J/3)
        
        # Finally, the (not necessarily ordered) eigenvalues
        t1 = 2*I*K - b/3
        t2 = -I*(K+L) - b/3
        t3 = -I*(K-L) - b/3
        
        # Order the eigenvalues using conditionals
        #
        T1_temp = conditional(lt(t1, t3), t3, t1 ) # returns the larger of t1 and t3.
        T1 = conditional(lt(T1_temp, t2), t2, T1_temp ) # returns the larger of T1_temp and t2.
        #
        T3_temp = conditional(gt(t3, t1), t1, t3 ) # returns the smaller of t1 and t3.
        T3 = conditional(gt(T3_temp, t2), t2, T1_temp ) # returns the smaller of T3_temp and t2.
        #
        # use the trace to report the middle eigenvalue.
        T2 = I1 - T1 - T3
        
        return T1, T2, T3


class ThemoElasticPlaneStrain():
    def __init__(self,u,p,theta,u_old,p_old,theta_old,domain,**kwargs):
        self.k_B      = Constant(domain,1.38E-17)             # Boltzmann's constant
        self.theta0   = Constant(domain,298.0)                  # Initial temperature
        self.Gshear_0 = Constant(domain,280.0)                  # Ground sate shear modulus
        self.N_R      = Constant(domain,PETSc.ScalarType(self.Gshear_0/(self.k_B*self.theta0)))# Number polymer chains per unit ref. volume
        self.lambdaL  = Constant(domain,5.12)                 # Locking stretch
        self.Kbulk    = Constant(domain,PETSc.ScalarType(1000*self.Gshear_0))        # Bulk modulus
        self.alpha    = Constant(domain,180.0E-6)             # Coefficient of thermal expansion
        self.c_v      = Constant(domain,1930.0)                 # Specific heat
        self.k_therm  = Constant(domain,0.16E3)               # Thermal conductivity

        if "alpha" in kwargs.keys():
            self.alpha = kwargs["alpha"]

        self.u=u
        self.p =p
        self.theta= theta

        self.u_old=u_old
        self.p_old =p_old
        self.theta_old= theta_old
        self.x = SpatialCoordinate(domain)
        self.domain = domain
    
    def Kinematics(self, u= None,p=None,theta=None,u_old= None,p_old=None,theta_old=None):
        if u == None:
            u = self.u
        if p == None:
            p = self.p
        if theta == None:
            theta = self.theta
        if u_old == None:
            u_old = self.u_old
        if p_old == None:
            p_old = self.p_old
        if theta_old == None:
            theta_old = self.theta_old
        
        self.F = self.F_pe_calc(u)
        J = det(self.F)   
        #
        self.lambdaBar = self.lambdaBar_calc(u)
        #
        self.F_old = self.F_pe_calc(u_old)
        self.J_old = det(self.F_old)   
        #
        self.C     = self.F.T*self.F
        self.C_old = self.F_old.T*self.F_old

        #  Tmat stress
        Tmat = self.Tmat_calc(u, p, theta)

        # Calculate the stress-temperature tensor
        self.M = self.M_calc(u)

        # Calculate the heat flux
        self.Qmat = self.Heat_flux_calc(u,theta)

        return Tmat,J
    
    def pe_grad_vector(self,u):
        grad_u = grad(u)
        return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                  [grad_u[1,0], grad_u[1,1], 0],
                  [0, 0, 0]]) 

    # Gradient of scalar field y
    # (just need an extra zero for dimensions to work out)
    def pe_grad_scalar(self,y):
        grad_y = grad(y)
        return as_vector([grad_y[0], grad_y[1], 0.])

    # Plane strain deformation gradient 
    def F_pe_calc(self,u):
        dim = len(u)
        Id = Identity(dim)          # Identity tensor
        F  = Id + grad(u)            # 2D Deformation gradient
        return as_tensor([[F[0,0], F[0,1], 0],
                    [F[1,0], F[1,1], 0],
                    [0, 0, 1]]) # Full pe F

    def lambdaBar_calc(self,u):
        F = self.F_pe_calc(u)
        J    = det(F)
        C    = F.T*F
        Cdis = J**(-2/3)*C
        I1   = tr(Cdis)
        lambdaBar = sqrt(I1/3.0)
        return lambdaBar

    def zeta_calc(self,u):
        lambdaBar = self.lambdaBar_calc(u)
        # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
        z    = lambdaBar/self.lambdaL
        z    = conditional(gt(z,0.95), 0.95, z) # Prevent the function from blowing up
        beta = z*(3.0 - z**2.0)/(1.0 - z**2.0)
        zeta = (self.lambdaL/(3*lambdaBar))*beta
        return zeta


    # Piola stress 
    def Tmat_calc(self,u, p, theta):
        F = self.F_pe_calc(u)
        J = det(F)
        #
        zeta = self.zeta_calc(u)
        Gshear  = self.N_R * self.k_B * theta * zeta
        #
        C  = F.T*F
        Tmat = J**(-2/3) * Gshear * (F - (1/3)*tr(C)*inv(F.T) ) - J * p * inv(F.T)
        return Tmat

    # Calculate the stress temperature tensor
    def M_calc(self,u):
        Id  = Identity(3)         
        F   = self.F_pe_calc(u) 
        #
        C  = F.T*F
        Cinv = inv(C) 
        J = det(F)
        zeta = self.zeta_calc(u)
        #
        fac1 = self.N_R * self.k_B * zeta
        fac2 = (3*self.Kbulk*self.alpha)/J
        #
        M =  J**(-2/3) * fac1 * (Id - (1/3)*tr(C)*Cinv)  - J * fac2 * Cinv
        return M
        
    #  Heat flux
    def Heat_flux_calc(self,u, theta):
        F = self.F_pe_calc(u) 
        J = det(F)         
        #
        Cinv = inv(F.T*F) 
        #
        Tcond = J * self.k_therm * Cinv # Thermal conductivity tensor
        #
        Qmat = - Tcond * self.pe_grad_scalar(theta)
        return Qmat

   
class ThermoElastic3D():
    def __init__(self,u,p,theta,u_old,p_old,theta_old,domain,**kwargs):
        self.k_B      = Constant(domain,1.38E-17)             # Boltzmann's constant
        self.theta0   = Constant(domain,298.0)                  # Initial temperature
        self.Gshear_0 = Constant(domain,280.0)                  # Ground sate shear modulus
        self.N_R      = Constant(domain,PETSc.ScalarType(self.Gshear_0/(self.k_B*self.theta0)))# Number polymer chains per unit ref. volume
        self.lambdaL  = Constant(domain,5.12)                 # Locking stretch
        self.Kbulk    = Constant(domain,PETSc.ScalarType(1000*self.Gshear_0))        # Bulk modulus
        self.alpha    = Constant(domain,180.0E-6)             # Coefficient of thermal expansion
        self.c_v      = Constant(domain,1930.0)                 # Specific heat
        self.k_therm  = Constant(domain,0.16E3)               # Thermal conductivity

        if "alpha" in kwargs.keys():
            self.alpha = kwargs["alpha"]

        self.u=u
        self.p =p
        self.theta= theta

        self.u_old=u_old
        self.p_old =p_old
        self.theta_old= theta_old
        self.x = SpatialCoordinate(domain)
        self.domain = domain

    def Kinematics(self, u= None,p=None,theta=None,u_old= None,p_old=None,theta_old=None):
            if u == None:
                u = self.u
            if p == None:
                p = self.p
            if theta == None:
                theta = self.theta
            if u_old == None:
                u_old = self.u_old
            if p_old == None:
                p_old = self.p_old
            if theta_old == None:
                theta_old = self.theta_old
            
            self.F = self.F_calc(u)
            J = det(self.F)   
            #
            self.lambdaBar = self.lambdaBar_calc(u)
            #
            self.F_old = self.F_calc(u_old)
            self.J_old = det(self.F_old)   
            #
            self.C     = self.F.T*self.F
            self.C_old = self.F_old.T*self.F_old

            #  Tmat stress
            Tmat = self.Tmat_calc(u, p, theta)

            # Calculate the stress-temperature tensor
            self.M = self.M_calc(u)

            # Calculate the heat flux
            self.Qmat = self.Heat_flux_calc(u,theta)

            return Tmat,J

    def F_calc(self,u):
        Id = Identity(3) 
        F  = Id + grad(u) 
        return F

    def lambdaBar_calc(self,u):
        F    = self.F_calc(u)
        J    = det(F)
        C    = F.T*F
        Cdis = J**(-2/3)*C
        I1   = tr(Cdis)
        lambdaBar = sqrt(I1/3.0)
        return lambdaBar

    def zeta_calc(self,u):
        lambdaBar = self.lambdaBar_calc(u)
        # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
        z    = lambdaBar/self.lambdaL
        z    = conditional(gt(z,0.95), 0.95, z) # Prevent the function from blowing up
        beta = z*(3.0 - z**2.0)/(1.0 - z**2.0)
        zeta = (self.lambdaL/(3*lambdaBar))*beta
        return zeta


    # Piola stress 
    def Tmat_calc(self,u, p, theta):
        F = self.F_calc(u)
        J = det(F)
        C  = F.T*F
        #
        zeta = self.zeta_calc(u)
        Gshear  = self.N_R * self.k_B * theta * zeta
        #
        Tmat = J**(-2/3) * Gshear * (F - (1/3)*tr(C)*inv(F.T) ) - J * p * inv(F.T)
        return Tmat

    # Calculate the stress temperature tensor
    def M_calc(self,u):
        Id  = Identity(3)         
        F   = self.F_calc(u) 
        #
        C  = F.T*F
        Cinv = inv(C) 
        J = det(F)
        zeta = self.zeta_calc(u)
        #
        fac1 = self.N_R * self.k_B * zeta
        fac2 = (3*self.Kbulk*self.alpha)/J
        #
        M =  J**(-2/3) * fac1 * (Id - (1/3)*tr(C)*Cinv)  - J * fac2 * Cinv
        return M
        
    #  Heat flux
    def Heat_flux_calc(self,u, theta):
        F = self.F_calc(u) 
        J = det(F)         
        #
        Cinv = inv(F.T*F) 
        #
        Tcond = J * self.k_therm * Cinv # Thermal conductivity tensor
        #
        Qmat = - Tcond * grad(theta)
        return Qmat