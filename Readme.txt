Read me
====================================

RAG Chatbot: Get Answers from Documents and Webpages
This chatbot, powered by OpenAI's large language model (LLM), can answer your questions based on what you give it!
What can it do?
    • Able to upload documents (PDF, DOCX, TXT) and accept a webpage URL (e.g., Confluence page).
    • Ask your question and get answers based on the uploaded content.

Getting Started:
Clone the repo
Perform following commands to run the code in virtual environment
• python -m venv ragenv
• source ragenv/bin/activate

export OPENAI_API_KEY="your open ai api key"

Install following libraries
pip install openai streamlit PyPDF2 docx requests-html langchain exception lxml_html_clean langchain-community chromadb tiktoken

streamlit run chatbot_app.py


Just install the openai library with pip install openai.
Known Issues:
    • Currently, the question field is in the middle of the page ideally should be at the bottom of the page.
    • Text entered in the question field doesn't automatically clear after submitting. User can edit and ask a new question.
Feel free to explore the code and contribute to the project!
