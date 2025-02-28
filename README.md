# Quantum NLP + GPT-4 chatbot using Streamlit

### **Functionality of the Code**
This Python script builds a **Quantum NLP + GPT-4 chatbot** using **Streamlit** for deployment. The chatbot combines quantum computing techniques with **GPT-4** to process user input and generate responses.

---

### **Breakdown of the Code**
#### **1️⃣ Importing Required Libraries**
- `streamlit` → Web app interface.
- `openai` → Access GPT-4 for chatbot responses.
- `numpy` → Used for numerical processing.
- `qiskit` & `qiskit_machine_learning` → Implements quantum feature maps and neural networks.

#### **2️⃣ OpenAI API Key Handling**
- The script assigns an **API key** directly (`openai.api_key`).
- If the key is missing, Streamlit shows an error and **stops execution**.

#### **3️⃣ Quantum NLP Embedding**
- Uses **Qiskit's `ZZFeatureMap`** to convert text into quantum representations.
- The **StatevectorEstimator** simulates quantum operations classically.
- **`SparsePauliOp("Z" * num_qubits)`** represents a Pauli-Z observable for quantum measurement.
- `EstimatorQNN` (Quantum Neural Network) is used for text feature extraction.

#### **4️⃣ Quantum Text Embedding Function**
- Converts user text into a **numerical quantum vector** (modulus 4 of ASCII).
- Passes the quantum-encoded text into the QNN for further transformation.

#### **5️⃣ GPT-4 Response Generator**
- Sends user input to GPT-4 using `openai.ChatCompletion.create()`.
- Includes a **system prompt** to guide GPT-4's responses.
- Returns the chatbot's reply.

#### **6️⃣ Streamlit Chatbot UI**
- Displays a **text input box** for the user to enter messages.
- Uses **quantum embedding** before passing the message to GPT-4.
- Displays the chatbot's response in a **text area**.

#### **7️⃣ Auto-Saving Code to `app.py`**
- The script automatically **writes itself to `app.py`**.
- This ensures the **correct structure** for deployment.

---

### **🚀 How This Works in Streamlit**
1. **User enters a message** → The input is converted to a quantum feature map.
2. **Quantum embedding modifies input** → GPT-4 processes both quantum & text features.
3. **GPT-4 generates a response** → Streamlit displays it in the UI.

---

### **🛠️ Next Steps for Deployment**
To **deploy** this on Streamlit Cloud:
1. Push this script to **GitHub**.
2. Ensure `requirements.txt` includes:
   ```
   streamlit
   openai
   numpy
   qiskit
   qiskit-machine-learning
   ```
3. Deploy the app via **Streamlit Cloud**.

### Contributors
1. **Aishika Das**.
2. **Devika Kanchan Simlai**
