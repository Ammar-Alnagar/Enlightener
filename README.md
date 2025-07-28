# Enlightener V2 :milky_way:

Welcome to Enlightener V2, a powerful and flexible Retrieval-Augmented Generation (RAG) framework. This framework is designed to be modular and easy to use, allowing you to build and experiment with different RAG pipelines.

## :rocket: Features

- **Modular Architecture**: Easily swap out components like embedding models, vector stores, and language models.
- **Configuration-driven**: All parameters are managed through a single `config.yaml` file.
- **Qdrant Integration**: Uses Qdrant as the vector store for efficient and scalable similarity search.
- **Contextual Compression**: Improves retrieval accuracy by using a language model to extract relevant information from the retrieved documents.
- **Extensible**: The framework is designed to be easily extended with new components and features.

## :building_construction: Project Structure

The project is organized as follows:

```
.
├── config/
│   └── config.yaml
├── core/
│   ├── __init__.py
│   ├── indexing.py
│   └── rag_framework.py
├── data/
│   └── ...
├── evaluation/
│   └── ...
├── notebooks/
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── test_indexing.py
│   └── test_rag_framework.py
├── utils/
│   └── ...
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

- **`config/`**: Contains the configuration files for the framework.
- **`core/`**: Contains the main application logic.
- **`data/`**: Contains the data to be indexed.
- **`evaluation/`**: Contains scripts for evaluating the performance of the RAG framework.
- **`notebooks/`**: Contains Jupyter notebooks for exploratory data analysis and model experimentation.
- **`tests/`**: Contains unit and integration tests.
- **`utils/`**: Contains helper functions.

## :gear: Getting Started

### Prerequisites

- Python 3.8+
- Pip
- Git

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/enlightener.git
   cd enlightener
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the environment variables:**

   Create a `.env` file in the root of the project and add the following environment variables:

   ```
   QDRANT_API_KEY="your_qdrant_api_key"
   QDRANT_URL="your_qdrant_url"
   GROQ_API_KEY="your_groq_api_key"
   ```

### Configuration

The framework is configured using the `config/config.yaml` file. You can modify this file to change the parameters of the RAG pipeline, such as the embedding model, the language model, and the retriever settings.

### Indexing

To index your data, run the following command:

```bash
python core/indexing.py
```

This will load the documents from the `data/` directory, split them into chunks, create embeddings, and store them in the Qdrant vector store.

### Usage

To use the RAG framework, you can run the `rag_framework.py` script:

```bash
python core/rag_framework.py
```

This will start an interactive session where you can ask questions to the RAG pipeline.

## :chart_with_upwards_trend: Architecture

The RAG framework is composed of the following components:

- **Data Loader**: Loads documents from various sources, such as PDF files.
- **Text Splitter**: Splits the documents into smaller chunks.
- **Embedding Model**: Creates embeddings for the text chunks.
- **Vector Store**: Stores the embeddings and allows for efficient similarity search.
- **Retriever**: Retrieves the most relevant documents from the vector store based on the user's query.
- **Language Model**: Generates a response based on the user's query and the retrieved documents.

Here is a diagram of the architecture:

```
[Architecture Diagram Placeholder]
A diagram showing the flow of data from the data loader to the language model.
```

## :heavy_check_mark: Testing

To run the tests, use the following command:

```bash
python -m unittest discover tests
```

## :handshake: Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any ideas for improvement.

## :scroll: License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
