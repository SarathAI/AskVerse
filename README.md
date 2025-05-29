# AskVerse
AskVerse: AI Conversation Agent for Smarter Student Learning is an AI-powered system designed to assist students by acting as an intelligent learning companion

🧠 AskVerse - Streamlit LLM App
This project sets up a local development environment for running an LLM-based application using Streamlit, LangChain, OpenAI API, and other supporting tools.

📦 Installation Guide
Step 1: Install Ubuntu on Windows
Download the Ubuntu app from the Microsoft Store (WSL).

Step 2: Download Miniconda
bash
Copy
Edit
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
Step 3: Install Miniconda
bash
Copy
Edit
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh
Follow the on-screen instructions to complete the installation.

Step 4: Update Conda
bash
Copy
Edit
conda update conda
Step 5: Create a Conda Environment (Python 3.8)
bash
Copy
Edit
conda create -n env_name python=3.8
Step 6: Install Required System Packages
bash
Copy
Edit
sudo apt install python3-dev
sudo apt-get install build-essential -y
Step 7: Install Python Dependencies
bash
Copy
Edit
pip install langchain unstructured openai chromadb Cython tiktoken pypdf streamlit --user
Step 8: (Optional) Repeat for Python 3.7 Environment
bash
Copy
Edit
conda create -n env_name_37 python=3.7
# Repeat Steps 6 and 7 inside this environment
Step 9: Install Additional Python Libraries
bash
Copy
Edit
pip install "transformers[torch]"
pip install scikit-learn stqdm flask
📚 Register Environment with Jupyter Kernel
bash
Copy
Edit
python -m ipykernel install --user --name=llm_env
🚀 Running the Streamlit App
1. Add Your OpenAI API Key
Replace your API key in the .streamlit/secrets.toml file:

toml
Copy
Edit
[openai]
api_key = "your_openai_api_key_here"
2. Navigate to App Directory
bash
Copy
Edit
cd "your/project/folder/path"
3. Run the App
bash
Copy
Edit
streamlit run AskVerse.py
🛠️ Tech Stack
Python 3.8 / 3.7

Streamlit

LangChain

OpenAI API

ChromaDB

Transformers

Flask

📄 License
This project is licensed under the MIT License.
