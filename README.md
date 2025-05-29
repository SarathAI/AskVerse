🧠 AskVerse - Streamlit LLM App

A Streamlit-based interactive AI app powered by OpenAI, LangChain, and ChromaDB for intelligent document Q&A and chatbot experiences.


📝 Table of Contents

About
Features
Installation
Setup Instructions
Usage
Configuration
Tech Stack
Screenshots
Contributing
License
Acknowledgements


📌 About
AskVerse is a local large language model app built with Python and Streamlit. It allows users to interact with documents and generate intelligent responses using the OpenAI API, LangChain pipelines, and ChromaDB vector storage.

✨ Features

🧠 Chat with LLMs using OpenAI API
📄 Ingest and analyze PDF or unstructured documents
🗃️ Use ChromaDB for document embedding
📊 Clean Streamlit-based UI for input/output
🛠️ Switch between Python 3.7 and 3.8 environments


⚙️ Installation
📥 Install Ubuntu (WSL)
Install the Ubuntu app from the Microsoft Store on your Windows system.
🐍 Download & Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh

Follow the prompts to complete installation and restart the terminal if needed.
🔄 Update Conda
conda update conda


🚀 Setup Instructions
✅ Create Conda Environment (Python 3.8)
conda create -n llm_env python=3.8
conda activate llm_env

🔧 Install System Dependencies
sudo apt install python3-dev
sudo apt-get install build-essential -y

📦 Install Python Libraries
pip install langchain unstructured openai chromadb Cython tiktoken pypdf streamlit --user

🧪 (Optional) Create Python 3.7 Env
conda create -n llm_env37 python=3.7

Repeat system dependency and pip install steps inside this env if needed.
🧠 Install ML & App Frameworks
pip install "transformers[torch]"
pip install scikit-learn stqdm flask

🧠 Register Environment with Jupyter Kernel
If you plan to use Jupyter with this environment:
python -m ipykernel install --user --name=llm_env


📦 Usage
🔑 Add Your OpenAI API Key
Create or edit the following file:
# .streamlit/secrets.toml
[openai]
api_key = "your_openai_api_key_here"

🚀 Run the App
Navigate to your project folder and start the app:
cd "your/project/folder/path"
streamlit run AskVerse.py

Open your browser and go to http://localhost:8501

🔐 Configuration
You can store additional secrets or config in .streamlit/secrets.toml:
[openai]
api_key = "sk-XXXX"

[custom]
project_name = "AskVerse"


🧰 Tech Stack

Python 3.8 / 3.7
Streamlit
LangChain
OpenAI API
ChromaDB
Transformers
Flask
scikit-learn
WSL + Ubuntu


📸 Screenshots
Add screenshots here after app runs successfully:
![AskVerse UI](screenshots/app_ui.png)


🤝 Contributing
Contributions are welcome! Follow these steps:

Fork the repository
Create your feature branch: git checkout -b feature/new-feature
Commit your changes: git commit -am 'Add new feature'
Push to the branch: git push origin feature/new-feature
Create a pull request


📄 License
This project is licensed under the MIT License. See the LICENSE file for more info.

🙏 Acknowledgements

Streamlit
LangChain
OpenAI
Hugging Face
ChromaDB

