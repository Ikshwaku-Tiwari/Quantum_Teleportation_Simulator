import streamlit as st
import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def stateVector_to_block(sv):
    alpha = sv.data[0]
    beta = sv.data[1]

    prob_0 = np.abs(alpha)**2

    theta = 2 * np.arccos(np.sqrt(prob_0))
    phi = np.angle(beta) - np.angle(alpha)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return [x, y, z]


