from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
    model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")

if __name__ == "__main__":
    download_model()