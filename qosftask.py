import numpy as np
import time
import matplotlib.pyplot as plt
I = np.array([[1, 0], [0, 1]], dtype=complex)  # Identity
X = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli-X (NOT gate)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)  # Hadamard

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)
def apply_gate(gate, state, target_qubit, n_qubits):
    full_gate = 1
    for i in range(n_qubits):
        if i == target_qubit:
            full_gate = np.kron(full_gate, gate)
        else:
            full_gate = np.kron(full_gate, I)
    return full_gate @ state
def apply_cnot(state, control_qubit, target_qubit, n_qubits):
    if control_qubit == 0 and target_qubit == 1:
        full_gate = CNOT
    else:
        full_gate = 1
        for i in range(n_qubits - 2):
            if i == control_qubit:
                full_gate = np.kron(full_gate, I)
            elif i == target_qubit:
                full_gate = np.kron(full_gate, CNOT)
            else:
                full_gate = np.kron(full_gate, I)
def simulate_circuit(n_qubits):
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1  

    start_time = time.time()
    
    for qubit in range(n_qubits):
        state = apply_gate(H, state, qubit, n_qubits)
    
    # apply CNOT between the first and second qubits
    if n_qubits > 1:
        state = apply_cnot(state, 0, 1, n_qubits)
    
    end_time = time.time()
    return end_time - start_time
qubits = list(range(1, 10))  
runtimes = []
for n in qubits:
    runtime = simulate_circuit(n)
    runtimes.append(runtime)
    print(f"{n} qubits: {runtime:.5f} seconds")
plt.plot(qubits, runtimes, marker='o')
plt.xlabel("Number of Qubits")
plt.ylabel("Runtime (seconds)")
plt.title("Quantum Circuit Simulation Runtime")
plt.yscale("log")
plt.show()
