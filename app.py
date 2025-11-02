import streamlit as st
import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import Statevector, partial_trace, Operator
from qiskit.circuit.library.standard_gates import XGate, ZGate
from qiskit_aer import AerSimulator 

PAULI_X = Operator([[0, 1], [1, 0]])
PAULI_Y = Operator([[0, -1j], [1j, 0]])
PAULI_Z = Operator([[1, 0], [0, -1]])

def get_bloch_coordinates(statevector, q_index):
    num_qubits = statevector.num_qubits
    qubits_to_trace_out = list(range(num_qubits))
    qubits_to_trace_out.pop(q_index)
    
    rho_qubit = partial_trace(statevector, qubits_to_trace_out)
    
    x = rho_qubit.expectation_value(PAULI_X).real
    y = rho_qubit.expectation_value(PAULI_Y).real
    z = rho_qubit.expectation_value(PAULI_Z).real
    
    return [x, y, z]

def create_bloch_sphere(vector, title):
    sphere = go.Surface(
        x=np.outer(np.cos(np.linspace(0, 2 * np.pi, 30)), np.sin(np.linspace(0, np.pi, 30))),
        y=np.outer(np.sin(np.linspace(0, 2 * np.pi, 30)), np.sin(np.linspace(0, np.pi, 30))),
        z=np.outer(np.ones(30), np.cos(np.linspace(0, np.pi, 30))),
        opacity=0.2,
        showscale=False,
        colorscale=[[0, 'rgb(100,100,100)'], [1, 'rgb(100,100,100)']],
        name="Sphere"
    )
    arrow = go.Scatter3d(
        x=[0, vector[0]],
        y=[0, vector[1]],
        z=[0, vector[2]],
        mode='lines',
        line=dict(color='red', width=10),
        name='State Vector'
    )
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', range=[-1, 1], showticklabels=False, showgrid=False),
            yaxis=dict(title='Y', range=[-1, 1], showticklabels=False, showgrid=False),
            zaxis=dict(title='Z (|0⟩ / |1⟩)', range=[-1, 1], showticklabels=True),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig = go.Figure(data=[sphere, arrow], layout=layout)
    return fig

st.set_page_config(layout="wide")
st.title("👻 Eídōlon (Phantom): The Teleportation Simulator")
st.markdown("An interactive 3-step visualization of the 2022 Nobel Prize-winning quantum teleportation protocol.")

st.header(f"1. Create Alice's 'Message' Qubit ($q_0$)")
theta_input = st.slider(
    "Set the superposition angle (θ):",
    min_value=0.0,
    max_value=np.pi,
    value=np.pi / 2,
    format="%.2f rad"
)
st.latex(rf"|\psi\rangle = \cos({theta_input/2:.2f})|0\rangle + \sin({theta_input/2:.2f})|1\rangle")

st.divider()

st.header("2. The Simulation")
st.markdown("Follow the state of the three qubits from left to right.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Step 1: Initial State")
    
    before_qc = QuantumCircuit(3)
    before_qc.ry(theta_input, 0)
    before_sv = Statevector(before_qc)
    
    q0_coords = get_bloch_coordinates(before_sv, 0)
    q1_coords = get_bloch_coordinates(before_sv, 1)
    q2_coords = get_bloch_coordinates(before_sv, 2)
    
    st.plotly_chart(create_bloch_sphere(q0_coords, "q₀ (Alice's Message)"), width='stretch')
    st.plotly_chart(create_bloch_sphere(q1_coords, "q₁ (Alice's Link)"), width='stretch')
    st.plotly_chart(create_bloch_sphere(q2_coords, "q₂ (Bob's Qubit)"), width='stretch')
    st.caption("Alice has her message ($q_0$). $q_1$ and $q_2$ are blank.")

with col2:
    st.subheader("Step 2: Just Before Measurement")
    
    mid_qc = QuantumCircuit(3, 2)
    mid_qc.ry(theta_input, 0)
    mid_qc.h(1)
    mid_qc.cx(1, 2)
    mid_qc.barrier()
    mid_qc.cx(0, 1)
    mid_qc.h(0)
    mid_qc.barrier()
    
    mid_sv = Statevector(mid_qc)
    
    q0_coords = get_bloch_coordinates(mid_sv, 0)
    q1_coords = get_bloch_coordinates(mid_sv, 1)
    q2_coords = get_bloch_coordinates(mid_sv, 2)
    
    st.plotly_chart(create_bloch_sphere(q0_coords, "q₀ (Entangled)"), width='stretch')
    st.plotly_chart(create_bloch_sphere(q1_coords, "q₁ (Entangled)"), width='stretch')
    st.plotly_chart(create_bloch_sphere(q2_coords, "q₂ (Broken State)"), width='stretch')
    st.caption("Alice performs her operations, entangling all 3 qubits. The states of $q_0$ and $q_1$ are now 'destroyed' (mixed) and $q_2$ is in a 'broken' state.")

with col3:
    st.subheader("Step 3: Bob Corrects")
    
    qr = QuantumRegister(3, name="q")
    cr_z = ClassicalRegister(1, name="cr_z")
    cr_x = ClassicalRegister(1, name="cr_x")
    after_qc = QuantumCircuit(qr, cr_z, cr_x)

    after_qc.ry(theta_input, 0)
    after_qc.h(1)
    after_qc.cx(1, 2)
    after_qc.barrier()
    after_qc.cx(0, 1)
    after_qc.h(0)
    after_qc.barrier()
    
    after_qc.measure([0, 1], [1, 0])
    after_qc.barrier()

    
    with after_qc.if_test((cr_x, 1)) as else_:
        after_qc.x(2)
    with after_qc.if_test((cr_z, 1)) as else_:
        after_qc.z(2)
    
    

    after_qc.save_statevector()
    
    simulator = AerSimulator()
    result = simulator.run(after_qc).result()
    final_sv = result.get_statevector()
    
    
    q0_coords = get_bloch_coordinates(final_sv, 0)
    q1_coords = get_bloch_coordinates(final_sv, 1)
    q2_coords = get_bloch_coordinates(final_sv, 2)

    st.plotly_chart(create_bloch_sphere(q0_coords, "q₀ (Collapsed)"), width='stretch')
    st.plotly_chart(create_bloch_sphere(q1_coords, "q₁ (Collapsed)"), width='stretch')
    st.plotly_chart(create_bloch_sphere(q2_coords, "q₂ (Teleported State)"), width='stretch')
    st.caption("Bob gets Alice's 2 classical bits, applies his correction, and recovers the original message. Teleportation complete!")
    
st.divider()
st.header("3. The Full Quantum Circuit")
st.text("This is the complete Qiskit circuit that runs in 'Step 3'.")
st.text(after_qc.draw(output='text'))