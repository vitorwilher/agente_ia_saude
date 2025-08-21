# AI Health Agent

This project is an AI-powered health agent that can answer questions based on a collection of documents. It uses a Retrieval Augmented Generation (RAG) model to provide accurate and context-aware answers.

## Features

*   **Document-based Q&A:** The agent can answer questions based on the documents provided in the `database` directory.
*   **RAG Model:** Utilizes a RAG model to retrieve relevant information from the documents and generate human-like answers.
*   **Configurable:** The agent's behavior can be configured through the `configs.json` file.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/analisemacropro/agente_ia_saude.git
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up the environment variables:**
    Create a `.env` file in the root directory and add the following:
    ```
    GOOGLE_API_KEY="your_gemini_api_key"
    ```

## Usage

To run the application, execute the following command:

```bash
shiny run src/app.py
```

## Configuration

The `configs.json` file contains the configuration for the app. You can modify this file to change the app's behavior.

## Dependencies

The dependencies for this project are listed in the `requirements.txt` file.
