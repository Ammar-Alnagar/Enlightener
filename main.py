import yaml
from src.data_processing import load_and_preprocess_data
from src.mamba_model import MambaModel
from src.retriever import Retriever
from src.generator import Generator


def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load and preprocess data
    data = load_and_preprocess_data(config['data_path'])

    # Initialize Mamba model
    mamba_model = MambaModel(config)

    # Encode documents
    embeddings = [mamba_model.encode(doc) for doc in data]

    # Initialize retriever
    retriever = Retriever(embeddings)




    # Initialize generator
    generator = Generator(config)

    # Example query
    query = "What is the capital of France?"
    query_embedding = mamba_model.encode(query)

    # Retrieve relevant documents
    relevant_doc_indices = retriever.retrieve(query_embedding)
    relevant_docs = [data[i] for i in relevant_doc_indices]

    # Generate response
    prompt = f"Query: {query}\nRelevant information: {' '.join(relevant_docs)}\nAnswer:"
    response = generator.generate(prompt)



    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()

# configs/config.yaml
data_path: 'data/processed/corpus.csv'
model_name: 'state-spaces/mamba-2.8b'
generator_model: 'gpt2'
