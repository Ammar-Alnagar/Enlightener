from models.mamba_model import MambaModel
from utils.data_processing import preprocess_data
from utils.evaluation import evaluate_model

def main():
    _, _, test_data = preprocess_data()
    vocab_size = 30000  # Example value, adjust as needed
    d_model = 512
    n_layers = 6
    model = MambaModel(vocab_size, d_model, n_layers)
    
    # Load trained model weights
    model.load_state_dict(torch.load('models/trained_mamba_model.pth'))
    
    results = evaluate_model(model, test_data)
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()
