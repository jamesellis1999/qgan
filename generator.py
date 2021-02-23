from torch import nn
import torch
import pennylane as qml
import numpy as np
from circuit import QuantumSim
import time

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator_Quantum(nn.Module):
    def __init__(self, n_qubits, q_depth, q_delta=0.1):
        """
        This is the quantum generator as described in https://arxiv.org/pdf/2010.06201.pdf
        """
        super().__init__()
        self.q_params = nn.ParameterList([nn.Parameter(q_delta * torch.randn(q_depth * n_qubits)) for i in range(8)])
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        # Spread of the random parameters for the paramaterised quantum gates
        self.q_delta = q_delta
      
        device = qml.device('lightning.qubit', wires=self.n_qubits)
        self.quantum_sim = QuantumSim(n_qubits, q_depth)

        self.qnodes = qml.QNodeCollection(
            [qml.QNode(self.quantum_sim.circuit, device, interface="torch") for i in range(8)]
        )
   
    def forward(self, noise):

        q_out = torch.Tensor(0, 8* (2**self.n_qubits))
        q_out = q_out.to(device)

        # Apply the quantum circuit to each element of the batch and append to q_out
        for elem in noise:
            
            patched_array = np.empty((0, 2**self.n_qubits))    
            for p, qnode in zip(self.q_params, self.qnodes):
                q_out_elem = qnode(elem, p).float().detach().cpu().numpy()
                patched_array = np.append(patched_array, q_out_elem)
            
            patched_tensor = torch.Tensor(patched_array).to(device).reshape(1, 8* (2**self.n_qubits))  
            q_out = torch.cat((q_out, patched_tensor))

        return q_out
