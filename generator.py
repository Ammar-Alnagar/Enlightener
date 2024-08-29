from transformers import AutoModelForCausalLM, AutoTokenizer

class Generator:
    def __init__(self, config):
        self.model = AutoModelForCausalLM.from_pretrained(config['generator_model'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['generator_model'])

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
