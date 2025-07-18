# RAG-API: Retrieval Augmented Generation API

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Acceptable Usage Policy](#acceptable-usage-policy)

## Introduction
This codebase is a Retrieval Augmented Generation (RAG) API that leverages the power of large language models (LLMs) and vector databases to provide accurate and relevant answers to user queries. It is designed to handle queries related to specific domains, such as Finance and Oil & Gas, by retrieving and synthesizing information from a knowledge base of documents.


## Features
- **Query Processing**: The API can handle user queries related to the healthcare. It performs query understanding, document retrieval, and answer generation using state-of-the-art language models and vector databases.

- **Document Retrieval**: The codebase utilizes the LlamaIndex library to index and store documents in a vector database (Qdrant). It employs efficient vector similarity search to retrieve the most relevant documents for a given query.

- **Answer Generation**: The API generates human-readable answers by synthesizing information from the retrieved documents. It uses large language models to generate coherent and context-aware responses.

- **Citation Support**: The API cites the relevant sources used to generate the answer, providing transparency and credibility to the information presented.

- **Related Queries**: In addition to the main answer, the API suggests related queries that users might be interested in exploring further.

- **Conversation History**: The API maintains a conversation history for each unique chat id, allowing for context-aware responses and better continuity in the dialogue.

- **Deployment and Monitoring**: The codebase is designed to be deployed on AWS infrastructure, with support for logging and monitoring using CloudWatch.

- **Modular Design**: The codebase follows a modular design, making it easy to extend, maintain, and customize various components, such as prompt templates, vector stores, and language models.


## Prerequisites
- Python 3.8 or higher
- AWS Account (for deployment and storage)
- OpenAI API Key (for GPT-4 and GPT-4-Turbo models)
- Anthropic API Key (for Claude-3-Sonnet model)
- Cohere API Key (for reranking)
- Tavily API Key (for web search)
- Qdrant Vector Database (for document storage and retrieval)


## Installation Guide
1. Clone the repository:

```
git clone <repo link>
```

2. Navigate to the project directory:

```
cd retrieval_api
```

3. Create a virtual environment and activate it:

```
python3 -m venv venv
source venv/bin/activate
```

4. Install the required dependencies:


```
if ! command -v cargo &> /dev/null; then
    echo "Rust is not installed. Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi
```

```
source $HOME/.cargo/env
```

```
pip install -r requirements.txt
```

6. Update the configuration file with the appropriate values for your AWS resources, API keys, and other settings.

7. Store your API keys and other sensitive information in the AWS Secrets Manager (OPENAI_API_KEY, ANTHROPIC_API_KEY, COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY, TAVILY_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY).

8. Follow the Qdrant installation guide for your platform: https://qdrant.tech/documentation/install/ and start the Qdrant server.

## Docker Running instructions

### .env file
enter required secrets in .env as shown in env.example.


### Using Docker Compose

1. go to deployment folder
```sh
cd deployment
```

2. Running the Vector DB using Docker Compose
```sh
sudo docker compose up -d
```

3. Check whether Docker is healthy or not using this command:
```sh
sudo docker ps
```

4. Stopping Docker
```sh
sudo docker compose down
```


### Install Docker (If Not Already Installed)

1. Linux Installation:
```sh
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo systemctl status docker
```

#check if docker id is running
'''ps aux | grep dockerd'''
#If dockerd is not running, start it manually:
'''sudo dockerd'''


## Configuration
The application configuration is managed through the `config/config.yaml` file. Here are the main sections and their descriptions:
- HF_EMBED: The name of the Hugging Face embedding model to use for document embeddings.
- COHERE_RERANKER: The name of the Cohere reranker model to use for reranking retrieved documents.
- LLM_MODEL_TYPE: The type of language model to use (e.g., OPENAI, OLLAMA, ANTHROPIC).
- PROMPT_DIR: The directory where prompt templates are stored.
- OPENAI, OLLAMA, ANTHROPIC: Configuration settings for the respective language models.
- PDF_BASE_URL: The base URL for accessing PDF documents.
- S3_LOGS_BUCKET, S3_CHAT_HISTORY, S3_PERSIST_BUCKET, S3_PERSIST_DIR: AWS S3 bucket and directory settings for logs, chat histories, and persistent data.
- SECRETS_MANAGER: The name of the AWS Secrets Manager secret containing your API keys and other sensitive information.
- CLOUDWATCH_LOG_GROUP, CLOUDWATCH_LOG_STREAM: CloudWatch log group and stream names for logging.
- QDRANT_ENABLE_HYBRID: Whether to enable hybrid search in Qdrant.
- FASTEMBED_SPARSE_MODEL: The name of the FastEmbed sparse model to use for document embeddings.
- RAG_SIMILARITY_CUTOFF, RAG_SIMILARITY_TOP_K, RAG_RERANKED_TOP_N, RAG_CITATION_CHUNK_SIZE, RAG_CITATION_CHUNK_OVERLAP, RAG_STREAMING: Configuration settings for the RAG system.
- AWS_REGION: The AWS region where your resources are located.


## Usage
1. Start the Qdrant server.

2. Run the FastAPI application:

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

3. Send a POST request to the `/v1/chat` endpoint with the following JSON payload:

```json
{
  "chat_id": "<unique_chat_id>",
  "query": "<user_query>",
  "category": "<category>",
  "file_name": "<optional_file_name>",
  "collection_name": "<optional_collection_name>",
  "persist_dir": "<optional_persist_directory>"
}
```

4. The API will return a streaming response with the following events:
   - **tokens**: Partial answer tokens as they are generated.
   - **answer**: The final answer to the query.
   - **context**: The context information (citations, sources) used to generate the answer.
   - **related**: Suggested related queries based on the original query and answer.
   - **title** (optional): A suggested title for the conversation thread.



## Project Structure
```
ratrieval-api/
├── config/
│   ├── config.yaml
│   └── config.yaml.example
├── prompts/
│   ├── citation_template.prompt
│   ├── conversation_title.prompt
│   ├── greeting.prompt
│   ├── greeting_classifier.prompt
│   ├── history_summarizer.prompt
│   ├── profanity_filter.prompt
│   ├── qa_template.prompt
│   ├── refine_template.prompt
│   ├── related_queries.prompt
│   ├── rephrased_query.prompt
│   ├── system.prompt
│   └── tavily.prompt
├── .gitignore
├── amazon-cloudwatch-agent.json
├── app.py
├── generate.py
├── README.md
├── requirements.txt
└── secrets_manager.py
```


## Troubleshooting
2. **OpenAI API Key Error**:
   - Verify that your OpenAI API key is correctly stored in the AWS Secrets Manager.
   - Check that you have sufficient quota and credits in your OpenAI account.

3. **Qdrant Connection Error**:
   - Ensure that the Qdrant server is running and accessible.
   - Check the Qdrant server logs for any errors or issues.
   - Verify that the Qdrant configuration in the `config.yaml` file is correct.

5. **Logging and Monitoring Issues**:
   - Ensure that the CloudWatch log group and stream names specified in the `config.yaml` file are correct.
   - Check the CloudWatch logs for any errors or issues.
   - Verify that your AWS credentials have the necessary permissions to access CloudWatch.

If you encounter any other issues or errors, please check the application logs for more information and consult the project documentation or seek support from the project maintainers.


## Acceptable Usage Policy

**This API is provided strictly for interview assessment and demonstration purposes. It is not intended for production, commercial, or any other use.**

Users are strictly prohibited from using the API for any purpose other than the interview assessment or demonstration for which access was granted. In particular, you may not:
- Use the API for any commercial, research, or production workloads.
- Engage in hate speech, discrimination, or harassment.
- Violate intellectual property rights or privacy laws.
- Attempt to bypass security measures or gain unauthorized access.

The API provider reserves the right to terminate access or take appropriate legal action against users who violate this policy.












