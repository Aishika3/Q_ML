import streamlit as st
import openai
import numpy as np
import os
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

# ----------------- Securely Load OpenAI GPT-4 API Key -----------------
openai.api_key = "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"  # Load API key from environment variables

if not openai.api_key:
    st.error("⚠️ OpenAI API key not found. Please set it as an environment variable.")
    st.stop()

# ----------------- Quantum NLP Embedding (Using StatevectorEstimator) -----------------
feature_map = ZZFeatureMap(feature_dimension=4)  # Quantum feature map
estimator = StatevectorEstimator()

# ✅ Define the observable (Pauli Z operator for measurement)
observable = SparsePauliOp("Z" * feature_map.num_qubits)  # SparsePauliOp for new Qiskit versions

# ✅ Correct argument order for `EstimatorQNN`
qnn = EstimatorQNN(
    circuit=feature_map,
    input_params=feature_map.parameters,
    observables=[observable],  # Pass as a list
    estimator=estimator
)

# ✅ Print the number of trainable weights
num_weights = qnn.num_weights
print(f"Number of trainable weights: {num_weights}")

# ----------------- Quantum Text Embedding Function -----------------
def quantum_text_embedding(text):
    """Convert text into a quantum numerical vector."""
    text_vector = np.array([ord(char) % 4 for char in text[:4]]) / 4  # Normalize input
    weights = np.random.rand(num_weights)  # ✅ Initialize random weights
    quantum_output = qnn.forward(text_vector, weights)  # ✅ Pass both inputs
    return quantum_output

# ----------------- GPT-4 Response Generator -----------------
def get_gpt4_response(user_input):
    """Generate a response using GPT-4."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful quantum AI chatbot."},
                {"role": "user", "content": user_input}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Error fetching GPT-4 response: {e}"

# ----------------- Streamlit Chatbot UI -----------------
st.title("🧠 Quantum NLP + GPT-4 Chatbot 🤖")

user_input = st.text_input("You: ")

if user_input:
    quantum_embedding = quantum_text_embedding(user_input)  # Get quantum representation
    modified_input = f"Quantum vector: {quantum_embedding.tolist()}, Message: {user_input}"

    gpt4_response = get_gpt4_response(modified_input)  # Send to GPT-4

    st.text_area("🤖 Chatbot:", value=gpt4_response, height=150, disabled=True)

# ✅ Save Streamlit App as a Python File
with open("app.py", "w") as f:
    f.write(
        "import streamlit as st\n"
        "import openai\n"
        "import numpy as np\n"
        "from qiskit.circuit.library import ZZFeatureMap\n"
        "from qiskit.primitives import StatevectorEstimator\n"
        "from qiskit.quantum_info import SparsePauliOp\n"
        "from qiskit_machine_learning.neural_networks import EstimatorQNN\n"
        "openai.api_key = 'your-api-key-here'\n"
        "feature_map = ZZFeatureMap(feature_dimension=4)\n"
        "estimator = StatevectorEstimator()\n"
        "observable = SparsePauliOp('Z' * feature_map.num_qubits)\n"
        "qnn = EstimatorQNN(\n"
        "    circuit=feature_map,\n"
        "    input_params=feature_map.parameters,\n"
        "    observables=[observable],\n"
        "    estimator=estimator\n"
        ")\n"
        "def quantum_text_embedding(text):\n"
        "    text_vector = np.array([ord(char) % 4 for char in text[:4]]) / 4\n"
        "    return qnn.forward(text_vector)\n"
        "def get_gpt4_response(user_input):\n"
        "    try:\n"
        "        response = openai.ChatCompletion.create(\n"
        "            model='gpt-4',\n"
        "            messages=[\n"
        "                {'role': 'system', 'content': 'You are a helpful quantum AI chatbot.'},\n"
        "                {'role': 'user', 'content': user_input}\n"
        "            ]\n"
        "        )\n"
        "        return response['choices'][0]['message']['content']\n"
        "    except Exception as e:\n"
        "        return f'⚠️ Error: {e}'\n"
        "def run():\n"
        "    st.title('🧠 Quantum NLP + GPT-4 Chatbot 🤖')\n"
        "    user_input = st.text_input('You: ')\n"
        "    if user_input:\n"
        "        quantum_embedding = quantum_text_embedding(user_input)\n"
        "        modified_input = f'Quantum vector: {quantum_embedding.tolist()}, Message: {user_input}'\n"
        "        gpt4_response = get_gpt4_response(modified_input)\n"
        "        st.text_area('🤖 Chatbot:', value=gpt4_response, height=150, disabled=True)\n"
        "if __name__ == '__main__':\n"
        "    run()\n"
    )
