# https://discuss.pennylane.ai/t/qnode-decorator-on-a-class-method-doesnt-work/100

import pennylane as qml
from circuit_layers import latent_layer, RY_layer, CZ_layer

class QuantumSim():
    def __init__(self, n_qubits, q_depth):
        self.n_qubits = n_qubits
        self.q_depth = q_depth 

    def circuit(self, noise, weights):
        weights = weights.reshape(self.q_depth, self.n_qubits)

        # Initialise latent vectors
        latent_layer(noise, self.n_qubits)

        # Depth of the circuit (1 by default)
        for i in range(self.q_depth):
            # Parameterised layer 
            RY_layer(self.n_qubits, weights[i])

            # Control Z gates
            CZ_layer(self.n_qubits)
    
        return qml.probs(wires=list(range(self.n_qubits)))   
      

