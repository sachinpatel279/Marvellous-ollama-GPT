MARVELLOUS AI - LOCAL STREAMLIT CHATBOT

Files:
- app.py: complete application
- assistant.jpg: assistant profile photo
- logo.png: logo asset with transparent white background
- chat_history.db: created automatically on first run
- generated_files/: generated output files are saved here automatically

1) Install base dependencies:
   py -3.12 -m pip install streamlit ollama pillow pypdf

2) Ensure Ollama is installed and running, then check installed models:
   ollama list

3) Text chat uses llama3.2:1b by default. If missing:
   ollama pull llama3.2:1b

4) Image analysis uses llava:7b by default. If missing:
   ollama pull llava:7b
   You can change the text and vision model names in the app sidebar.

5) Run:
   streamlit run app.py

Local image generation:
The app uses Diffusers with segmind/tiny-sd. Install image-generation dependencies separately:
   py -3.12 -m pip install torch diffusers transformers accelerate safetensors
The first image generation requires internet to download the model. Afterwards, the cached model can run locally. CPU generation can be slow and requires sufficient RAM/disk space.

Features:
- Futuristic Streamlit GUI
- Assistant avatar and top-left logo
- Searchable local chat history and delete controls
- Upload and analyze images/documents (PDF, TXT, CSV, Python and common code/text files)
- Local text-to-image generation
- Prompt-based downloadable file generation

Review AI-generated code before running it. Image understanding needs an Ollama vision model; a text-only model cannot analyze image content.
