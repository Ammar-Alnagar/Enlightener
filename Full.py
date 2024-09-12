# src/gguf_model.py
from llama_cpp import Llama
   
class GGUFModel:
    def __init__(self, model_path: str, n_ctx: int = 512, n_gpu_layers: int = 0):
        self.model = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)

    def encode(self, text: str) -> list:
        # Use the model to encode the text and return the last hidden state
        return self.model.embed(text)

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        output = self.model(prompt, max_tokens=max_tokens)
        return output['choices'][0]['text']

# src/utils.py
def determine_model_type(model_path: str) -> str:
    if model_path.endswith('.gguf'):
        return 'gguf'
    elif 'mamba' in model_path.lower():
        return 'mamba'
    else:
        return 'transformer'



# Update src/mamba_model.py
# (No changes needed if it's already implemented as shown earlier)

# Update src/generator.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.gguf_model import GGUFModel
from src.utils import determine_model_type

class Generator:
    def __init__(self, config):
        model_type = determine_model_type(config['generator_model'])
        if model_type == 'gguf':
            self.model = GGUFModel(config['generator_model'])
        else:
            self.model = AutoModelForCausalLM.from_pretrained(config['generator_model'])
            self.tokenizer = AutoTokenizer.from_pretrained(config['generator_model'])

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        if isinstance(self.model, GGUFModel):
            return self.model.generate(prompt, max_tokens=max_tokens)
        else:
            inputs = self.tokenizer(prompt, return_tensors='pt')
            outputs = self.model.generate(**inputs, max_length=max_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Update main.py
import yaml
from src.data_processing import load_and_preprocess_data
from src.mamba_model import MambaModel
from src.gguf_model import GGUFModel
from src.retriever import Retriever
from src.generator import Generator
from src.utils import determine_model_type

def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load and preprocess data
    data = load_and_preprocess_data(config['data_path'])

    # Initialize model based on type
    model_type = determine_model_type(config['model_name'])
    if model_type == 'gguf':
        model = GGUFModel(config['model_name'])
    elif model_type == 'mamba':
        model = MambaModel(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Encode documents
    embeddings = [model.encode(doc) for doc in data]

    # Initialize retriever
    retriever = Retriever(embeddings)

    # Initialize generator
    generator = Generator(config)

    # Example query
    query = "What is the capital of France?"
    query_embedding = model.encode(query)

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
data_path: 'data/processed/text'
model_name: 'state-spaces/mamba-1.4b'  # or path to GGUF model
generator_model: 'hermes3'  # or path to GGUF model

