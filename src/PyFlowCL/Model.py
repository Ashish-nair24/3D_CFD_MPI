"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Model.py

"""

__copyright__ = """
Copyright (c) 2022 University of Notre Dame
"""

__license__ = """
 Permission is hereby granted, free of charge, to any person 
 obtaining a copy of this software and associated documentation 
 files (the "Software"), to deal in the Software without 
 restriction, including without limitation the rights to use, 
 copy, modify, merge, publish, distribute, sublicense, and/or 
 sell copies of the Software, and to permit persons to whom the 
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be 
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
 WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
 OTHER DEALINGS IN THE SOFTWARE.
"""


import torch
import torch.nn as nn

class NeuralNetworkModel_ELU(nn.Module):
    def __init__(self, H, num_inputs, num_outputs, C_out,  alpha=1.0):
        super(NeuralNetworkModel_ELU, self).__init__()
        
        self.fc1 = nn.Linear(num_inputs, H).type(torch.DoubleTensor)
        self.fcG = nn.Linear(num_inputs, H).type(torch.DoubleTensor)
        
        self.fc2 = nn.Linear(H, H).type(torch.DoubleTensor)
        self.fc4 = nn.Linear(H, num_outputs).type(torch.DoubleTensor)
        
        self.C_out    = C_out
        self.alpha = alpha
        
    def forward(self, x):
        L1 =  self.fc1( x ) 
        
        #H1 = torch.relu( L1 )
        H1 = torch.relu(L1) - torch.relu(- self.alpha*( torch.exp(L1) - 1) )
        
        L2 = self.fc2( H1 ) 
        #H2 = torch.relu(L2 )
        #print(torch.exp(torch.max(L2)))
        H2 = torch.relu(L2) - torch.relu(- self.alpha*( torch.exp(L2) - 1) )
        L3 =  self.fcG( x ) 
        G = torch.sigmoid(L3)
        

        H2_G = G*H2
        
        f_out = self.fc4( H2_G )
        
        elu = torch.relu(f_out) - torch.relu( - self.alpha * ( torch.exp(f_out) - 1)) 

        return self.C_out * (elu)

class NeuralNetworkModel_tanh(nn.Module):
    def __init__(self, H, num_inputs_1, num_inputs_2, num_outputs, C_out):
        super(NeuralNetworkModel_tanh, self).__init__()
        
        self.fc1 = nn.Linear(num_inputs_1, H, bias=False).type(torch.DoubleTensor)
        self.fc2 = nn.Linear(num_inputs_2, H, bias=False).type(torch.DoubleTensor)
                
        self.fc3 = nn.Linear(H, H, bias=False).type(torch.DoubleTensor)
        self.fc4 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.fc5 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        
        self.C_out    = C_out
        self.alpha    = 1.0 #/ self.C_out
        
        self.H = H
        
    def forward(self, x, y):
        
        L1 = self.fc1(x)
        H1 = torch.tanh(L1)
        
        L2 = self.fc3(H1)
        H2 = torch.tanh(L2)
                
        L3 = self.fc2(y)
        H3 = torch.tanh(L3)
        
        (p1,q1) = torch.split(H2, int(self.H/2), dim=-1)
        (p2,q2) = torch.split(H3, int(self.H/2), dim=-1)
        
        f_out_1 = self.fc4(p1*p2)
        f_out_2 = self.fc5(q1*q2)
        
        elu_1 = torch.relu(f_out_1) - torch.relu( - self.alpha * ( torch.exp(f_out_1) - 1)) 
        elu_2 = torch.relu(f_out_2) - torch.relu( - self.alpha * ( torch.exp(f_out_2) - 1)) 
        
        # f_out = self.fc4(H3_H5)

        # elu = torch.relu(f_out) - torch.relu( - self.alpha * ( torch.exp(f_out) - 1)) 

        return torch.cat((elu_1,elu_2), dim=-1)

class NeuralNetworkModel_tanh_wm(nn.Module):
    def __init__(self, H, num_inputs_1, num_inputs_2, num_outputs, C_out):
        super(NeuralNetworkModel_tanh_wm, self).__init__()
        
        self.fc1 = nn.Linear(num_inputs_1, H, bias=False).type(torch.DoubleTensor)
        self.fc2 = nn.Linear(num_inputs_2, H, bias=False).type(torch.DoubleTensor)
                
        self.fc3 = nn.Linear(H, H, bias=False).type(torch.DoubleTensor)
        self.fc4 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.fc5 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)

        self.wm1 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.wm2 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.wm3 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.wm4 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)

        self.C_out    = C_out
        self.alpha    = 1.0 #/ self.C_out
        
        self.H = H
        
    def forward(self, x, y):
        
        L1 = self.fc1(x)
        H1 = torch.tanh(L1)
        
        L2 = self.fc3(H1)
        H2 = torch.tanh(L2)
                
        L3 = self.fc2(y)
        H3 = torch.tanh(L3)
        
        (p1,q1) = torch.split(H2, int(self.H/2), dim=-1)
        (p2,q2) = torch.split(H3, int(self.H/2), dim=-1)
        
        f_out_1 = self.fc4(p1*p2)
        f_out_2 = self.fc5(q1*q2)

        wm_out_1 = self.wm1(p1*p2)
        wm_out_2 = self.wm2(p1*p2)
        wm_out_3 = self.wm3(q1*q2)
        wm_out_4 = self.wm4(q1*q2)
        
        elu_1 = torch.relu(f_out_1) - torch.relu( - self.alpha * ( torch.exp(f_out_1) - 1)) 
        elu_2 = torch.relu(f_out_2) - torch.relu( - self.alpha * ( torch.exp(f_out_2) - 1)) 
        
        # f_out = self.fc4(H3_H5)

        # elu = torch.relu(f_out) - torch.relu( - self.alpha * ( torch.exp(f_out) - 1)) 

        return torch.cat((elu_1,elu_2,wm_out_1,wm_out_2,wm_out_3,wm_out_4), dim=-1)

class NeuralNetworkModel_ELU_dist_wall(nn.Module):
    def __init__(self, H, num_inputs, num_outputs, C_out,  alpha=0.95):
        super(NeuralNetworkModel_ELU_dist_wall, self).__init__()
        
        self.fc1 = nn.Linear(num_inputs, H).type(torch.DoubleTensor)
        self.fcG = nn.Linear(num_inputs, H).type(torch.DoubleTensor)
        
        self.fc2 = nn.Linear(H, H).type(torch.DoubleTensor)
        self.fc3 = nn.Linear(H, H).type(torch.DoubleTensor)
        
        self.fc4 = nn.Linear(H, num_outputs).type(torch.DoubleTensor)
        self.fc5 = nn.Linear(H, num_outputs).type(torch.DoubleTensor)
        self.fc6 = nn.Linear(H, num_outputs).type(torch.DoubleTensor)
        # self.fc7 = nn.Linear(H, num_outputs).type(torch.DoubleTensor)
        # self.fc8 = nn.Linear(H, num_outputs).type(torch.DoubleTensor)
        # self.fc9 = nn.Linear(H, num_outputs).type(torch.DoubleTensor)
        # self.fc10 = nn.Linear(H, num_outputs).type(torch.DoubleTensor)
        
        self.C_out    = C_out
        self.alpha = alpha
        
    def forward(self, x):
        L1 =  self.fc1( x ) 
        
        #H1 = torch.relu( L1 )
        H1 = torch.relu(L1) #- torch.relu(- self.alpha*( torch.exp(L1) - 1) )
        
        L2 = self.fc2( H1 ) 
        #H2 = torch.relu(L2 )
        #print(torch.exp(torch.max(L2)))
        H2 = torch.relu(L2) #- torch.relu(- self.alpha*( torch.exp(L2) - 1) )
        L3 = self.fc3( H2 ) 
        H3 = torch.relu(L3)
        
        L4 =  self.fcG( x ) 
        G = torch.sigmoid(L4)
        

        H2_G = G*H3
        
        # SG
        # f_out_1 = self.fc4( H2_G )
        # # f_out_1[:,1] = 1000. * (1 + torch.tanh(f_out_1[:,1]))
        # # f_out_1[:,3] = torch.relu(f_out_1[:,3])        
        # # f_out_1[:,4] = 1000. * (1 + torch.tanh(f_out_1[:,4]))
        # f_out_1[:,6] = 0.5 * (1. + torch.tanh(f_out_1[:,6]))
        
        # f_out_2 = self.fc5( H2_G )
        # # f_out_2[:,1] = 1000. * (1 + torch.tanh(f_out_2[:,1]))
        # # f_out_2[:,4] = 1000. * (1 + torch.tanh(f_out_2[:,4]))
        # f_out_2[:,6] = 0.5 * (1. + torch.tanh(f_out_2[:,6]))
        
        # f_out_3 = self.fc6( H2_G )
        # # f_out_3[:,1] = 1000. * (1 + torch.tanh(f_out_3[:,1]))
        # # f_out_3[:,4] = 1000. * (1 + torch.tanh(f_out_3[:,4]))
        # f_out_3[:,6] = 0.5 * (1. + torch.tanh(f_out_3[:,6]))
        
        # MB
        #m = torch.nn.functional.gelu()
        f_out_1 = self.fc4( H2_G )
        f_out_1[:,0] = 1000*((1. + (torch.relu(f_out_1[:,0]) + self.alpha * (torch.exp(-torch.relu(-f_out_1[:,0])) - 1))))#1000. * (1 + torch.tanh(f_out_1[:,0])) 
        # f_out_1[:,1] = 1000. * (1 + torch.tanh(f_out_1[:,1]))
        f_out_1[:,2] = 1000*((1. + (torch.relu(f_out_1[:,2]) + self.alpha * (torch.exp(-torch.relu(-f_out_1[:,2])) - 1))))#1000. * (1 + torch.tanh(f_out_1[:,2]))
        # f_out_1[:,3] = 1000. * (1 + torch.tanh(f_out_1[:,3]))
        # f_out_1[:,4] = 0.5 * (1. + torch.tanh(f_out_1[:,4]))
        f_out_1[:,4] = torch.sigmoid(f_out_1[:,4])
        
        f_out_2 = self.fc5( H2_G )
        f_out_2[:,0] = 1000*((1. + (torch.relu(f_out_2[:,0]) + self.alpha * (torch.exp(-torch.relu(-f_out_2[:,0])) - 1))))#1000. * (1 + torch.tanh(f_out_2[:,0]))
        # f_out_2[:,1] = 1000. * (1 + torch.tanh(f_out_2[:,1]))
        f_out_2[:,2] = 1000*((1. + (torch.relu(f_out_2[:,2]) + self.alpha * (torch.exp(-torch.relu(-f_out_2[:,2])) - 1))))#1000. * (1 + torch.tanh(f_out_2[:,2]))
        # f_out_2[:,3] = 1000. * (1 + torch.tanh(f_out_2[:,3]))
        #f_out_2[:,4] = 0.5 * (1. + torch.tanh(f_out_2[:,4]))
        f_out_2[:,4] = torch.sigmoid(f_out_2[:,4])

        f_out_3 = self.fc6( H2_G )
        f_out_3[:,0] = 1000*((1. + (torch.relu(f_out_3[:,0]) + self.alpha * (torch.exp(-torch.relu(-f_out_3[:,0])) - 1))))#1000. * (1 + torch.tanh(f_out_3[:,0]))
        # f_out_3[:,1] = 1000. * (1 + torch.tanh(f_out_3[:,1]))
        f_out_3[:,2] = 1000*((1. + (torch.relu(f_out_3[:,2]) + self.alpha * (torch.exp(-torch.relu(-f_out_3[:,2])) - 1))))#1000. * (1 + torch.tanh(f_out_3[:,2]))
        # f_out_3[:,3] = 1000. * (1 + torch.tanh(f_out_3[:,3]))
        # f_out_3[:,4] = 0.5 * (1. + torch.tanh(f_out_3[:,4]))
        f_out_3[:,4] = torch.sigmoid(f_out_3[:,4])

        # f_out_4 = self.fc7( H2_G )
        # f_out_5 = self.fc8( H2_G )
        # f_out_6 = self.fc9( H2_G )
        # f_out_7 = self.fc10( H2_G )
        # elu_1 = (f_out_1[:,:6]) #- torch.relu( - self.alpha * ( torch.exp(f_out_1[:,:6]) - 1)) 
        # elu_2 = (f_out_2[:,:6]) #- torch.relu( - self.alpha * ( torch.exp(f_out_2[:,:6]) - 1)) 
        # elu_3 = (f_out_3[:,:6]) #- torch.relu( - self.alpha * ( torch.exp(f_out_3[:,:6]) - 1)) 
        

        return torch.cat((f_out_1, f_out_2, f_out_3), dim=-1)#torch.cat((elu_1,f_out_1[:,6].unsqueeze(dim=1),elu_2,f_out_2[:,6].unsqueeze(dim=1),elu_3,f_out_3[:,6].unsqueeze(dim=1)), dim=-1)


class NeuralNetworkModel_tanh_aniso(nn.Module):
    def __init__(self, H, num_inputs_1, num_inputs_2, num_outputs, C_out):
        super(NeuralNetworkModel_tanh_aniso, self).__init__()
        
        self.fc1 = nn.Linear(num_inputs_1, H, bias=False).type(torch.DoubleTensor)
        self.fc2 = nn.Linear(num_inputs_2, H, bias=False).type(torch.DoubleTensor)
                
        self.fc3 = nn.Linear(H, H, bias=False).type(torch.DoubleTensor)
        self.fc4 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.fc5 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.fc6 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.fc7 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.fc8 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        
        self.C_out    = C_out
        self.alpha    = 1.0 #/ self.C_out
        
        self.H = H
        
    def forward(self, x, y):
        
        L1 = self.fc1(x)
        H1 = torch.tanh(L1)
        
        L2 = self.fc3(H1)
        H2 = torch.tanh(L2)
                
        L3 = self.fc2(y)
        H3 = torch.tanh(L3)
        
        (p1,q1) = torch.split(H2, int(self.H/2), dim=-1)
        (p2,q2) = torch.split(H3, int(self.H/2), dim=-1)
        
        f_out_1 = self.fc4(p1*p2)
        f_out_2 = self.fc5(p1*p2)
        f_out_3 = self.fc6(p1*p2)
        f_out_4 = self.fc7(q1*q2)
        f_out_5 = self.fc8(q1*q2)
        
        elu_1 = torch.relu(f_out_1) - torch.relu( - self.alpha * ( torch.exp(f_out_1) - 1)) 
        elu_2 = torch.relu(f_out_2) - torch.relu( - self.alpha * ( torch.exp(f_out_2) - 1)) 
        elu_3 = torch.relu(f_out_3) - torch.relu( - self.alpha * ( torch.exp(f_out_3) - 1)) 
        elu_4 = torch.relu(f_out_4) - torch.relu( - self.alpha * ( torch.exp(f_out_4) - 1)) 
        elu_5 = torch.relu(f_out_5) - torch.relu( - self.alpha * ( torch.exp(f_out_5) - 1)) 
        
        # f_out = self.fc4(H3_H5)

        # elu = torch.relu(f_out) - torch.relu( - self.alpha * ( torch.exp(f_out) - 1)) 

        return torch.cat((elu_1,elu_2,elu_3,elu_4,elu_5), dim=-1)

class NeuralNetworkModel_tanh_aniso_wm(nn.Module):
    def __init__(self, H, num_inputs_1, num_inputs_2, num_outputs, C_out):
        super(NeuralNetworkModel_tanh_aniso_wm, self).__init__()
        
        self.fc1 = nn.Linear(num_inputs_1, H, bias=False).type(torch.DoubleTensor)
        self.fc2 = nn.Linear(num_inputs_2, H, bias=False).type(torch.DoubleTensor)
                
        self.fc3 = nn.Linear(H, H, bias=False).type(torch.DoubleTensor)
        self.fc4 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.fc5 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.fc6 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.fc7 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.fc8 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)

        self.wm1 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.wm2 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.wm3 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.wm4 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)

        self.C_out    = C_out
        self.alpha    = 1.0 #/ self.C_out
        
        self.H = H
        
    def forward(self, x, y):
        
        L1 = self.fc1(x)
        H1 = torch.tanh(L1)
        
        L2 = self.fc3(H1)
        H2 = torch.tanh(L2)
                
        L3 = self.fc2(y)
        H3 = torch.tanh(L3)
        
        (p1,q1) = torch.split(H2, int(self.H/2), dim=-1)
        (p2,q2) = torch.split(H3, int(self.H/2), dim=-1)
        
        f_out_1 = self.fc4(p1*p2)
        f_out_2 = self.fc5(p1*p2)
        f_out_3 = self.fc6(p1*p2)
        f_out_4 = self.fc7(q1*q2)
        f_out_5 = self.fc8(q1*q2)

        wm_out_1 = self.wm1(p1*p2)
        wm_out_2 = self.wm2(p1*p2)
        wm_out_3 = self.wm3(q1*q2)
        wm_out_4 = self.wm4(q1*q2)

        
        elu_1 = torch.relu(f_out_1) - torch.relu( - self.alpha * ( torch.exp(f_out_1) - 1)) 
        elu_2 = torch.relu(f_out_2) - torch.relu( - self.alpha * ( torch.exp(f_out_2) - 1)) 
        elu_3 = torch.relu(f_out_2) - torch.relu( - self.alpha * ( torch.exp(f_out_3) - 1)) 
        elu_4 = torch.relu(f_out_2) - torch.relu( - self.alpha * ( torch.exp(f_out_4) - 1)) 
        elu_5 = torch.relu(f_out_2) - torch.relu( - self.alpha * ( torch.exp(f_out_5) - 1)) 
        
        # f_out = self.fc4(H3_H5)

        # elu = torch.relu(f_out) - torch.relu( - self.alpha * ( torch.exp(f_out) - 1)) 

        return torch.cat((elu_1,elu_2,elu_3,elu_4,elu_5,wm_out_1,wm_out_2,wm_out_3,wm_out_4), dim=-1)


class NeuralNetworkModel_tanh_aniso_tf(nn.Module):
    def __init__(self, H, num_inputs_1, num_inputs_2, num_outputs, C_out):
        super(NeuralNetworkModel_tanh_aniso_tf, self).__init__()
        
        self.fc1 = nn.Linear(num_inputs_1, H, bias=False).type(torch.DoubleTensor)
        self.fc2 = nn.Linear(num_inputs_2, H, bias=False).type(torch.DoubleTensor)
                
        self.fc3 = nn.Linear(H, H, bias=False).type(torch.DoubleTensor)
        self.fc4 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.fc5 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        #self.fc6 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.fc7 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        self.fc8 = nn.Linear(int(H/2), 1, bias=False).type(torch.DoubleTensor)
        
        self.C_out    = C_out
        self.alpha    = 1.0 #/ self.C_out
        
        self.H = H
        
    def forward(self, x, y):
        
        L1 = self.fc1(x)
        H1 = torch.tanh(L1)
        
        L2 = self.fc3(H1)
        H2 = torch.tanh(L2)
                
        L3 = self.fc2(y)
        H3 = torch.tanh(L3)
        
        (p1,q1) = torch.split(H2, int(self.H/2), dim=-1)
        (p2,q2) = torch.split(H3, int(self.H/2), dim=-1)
        
        f_out_1 = self.fc4(p1*p2)
        f_out_2 = self.fc5(p1*p2)
        #f_out_3 = self.fc6(p1*p2)
        f_out_4 = self.fc7(q1*q2)
        f_out_5 = self.fc8(q1*q2)
        
        elu_1 = torch.relu(f_out_1) - torch.relu( - self.alpha * ( torch.exp(f_out_1) - 1)) 
        elu_2 = torch.relu(f_out_2) - torch.relu( - self.alpha * ( torch.exp(f_out_2) - 1)) 
        #elu_3 = torch.relu(f_out_2) - torch.relu( - self.alpha * ( torch.exp(f_out_3) - 1)) 
        elu_4 = torch.relu(f_out_2) - torch.relu( - self.alpha * ( torch.exp(f_out_4) - 1)) 
        elu_5 = torch.relu(f_out_2) - torch.relu( - self.alpha * ( torch.exp(f_out_5) - 1)) 
        
        # f_out = self.fc4(H3_H5)

        # elu = torch.relu(f_out) - torch.relu( - self.alpha * ( torch.exp(f_out) - 1)) 

        return torch.cat((elu_1,elu_2,elu_4,elu_5), dim=-1)

# class NeuralNetworkModel_ELU(nn.Module):
#     def __init__(self, H, num_inputs, num_outputs, C_out, alpha=1.0):
#         super(NeuralNetworkModel_ELU, self).__init__()
        
#         self.fc1 = nn.Linear(num_inputs, H).type(torch.DoubleTensor)
#         self.fcG = nn.Linear(num_inputs, H).type(torch.DoubleTensor)
        
#         self.fc2 = nn.Linear(H, H).type(torch.DoubleTensor)
#         self.fc4 = nn.Linear(H, num_outputs).type(torch.DoubleTensor)
        
#         self.C_out = C_out
#         self.alpha = alpha
        
#     def forward(self, x):
        
#         L1 =  self.fc1( x ) 
        
#         #H1 = torch.relu( L1 )
#         H1 = torch.relu(L1) - torch.relu(- self.alpha*( torch.exp(L1) - 1) )
        
#         L2 = self.fc2( H1 ) 
        
#         #H2 = torch.relu(L2 )
#         H2 = torch.relu(L2) - torch.relu(- self.alpha*( torch.exp(L2) - 1) )
        
#         L3 =  self.fcG( x ) 
#         G = torch.sigmoid(L3)
        

#         H2_G = G*H2
        
#         f_out = self.C_out*self.fc4( H2_G )
#         #f_out_1 = torch.nn.functional.elu(f_out)
        
#         return f_out

# ----------------------------------------------
# Single-layer "constant" model
# ----------------------------------------------
class SingleLayerConstModel(nn.Module):
    def __init__(self, perturb, num_outputs, initial_theta=None):
        super(SingleLayerConstModel, self).__init__()

        if initial_theta is None:
            self.theta = nn.Parameter(torch.tensor(perturb * 2.0 *
                                                   (np.random.random_sample(num_outputs) - 0.5))) 
        else:
            self.theta = nn.Parameter(initial_theta)
            
    def forward(self):
        return self.theta
    
    
    
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]).type(torch.DoubleTensor))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        
        output = self.net(x)
        
        # output_1 = torch.nn.functional.sigmoid(output[:,:,0])
        # output_2 = torch.nn.functional.sigmoid(output[:,:,1])
        # output_3 = torch.nn.functional.elu(output[:,:,2]) + 1.1
        # output_4 = 3 * torch.nn.functional.sigmoid(output[:,:,3])
        # output_5 = torch.nn.functional.elu(output[:,:,4]) + 1.1
        # output_6 = torch.nn.functional.elu(output[:,:,5]) + 1.1

        output_1 = torch.nn.functional.sigmoid(output[:,0])
        output_2 = torch.nn.functional.sigmoid(output[:,1])
        output_3 = torch.nn.functional.elu(output[:,2]) + 1.1
        output_4 = 3 * torch.nn.functional.sigmoid(output[:,3])
        output_5 = torch.nn.functional.elu(output[:,4]) + 1.1
        output_6 = torch.nn.functional.elu(output[:,5]) + 1.1
        
        # output_1 = torch.nn.functional.elu(output[:,0]) + 1.1
        # output_2 = torch.nn.functional.elu(output[:,1]) + 1.1
        # output_3 = torch.nn.functional.elu(output[:,2]) + 1.1
        # output_4 = torch.nn.functional.elu(output[:,3]) + 1.1
        # output_5 = torch.nn.functional.elu(output[:,4]) + 1.1
        # output_6 = torch.nn.functional.elu(output[:,5]) + 1.1

        
        return torch.stack((output_1, output_2, output_3, output_4, output_5, output_6), dim=1)
