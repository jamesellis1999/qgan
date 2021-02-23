import pennylane as qml

# NOTE each sub generator needs to have the same latent vector input.. this is what they did in the paper 
def latent_layer(noise, n_wires):
    for i in range(n_wires):
        qml.RY(noise[i], wires=i)

def RY_layer(n_wires, weights):

    for i in range(n_wires):
        qml.RY(weights[i], wires=i)

def CZ_layer(n_wires):

    for i in range(n_wires-1):
        qml.CZ(wires=[i,i+1])