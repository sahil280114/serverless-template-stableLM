from potassium import Potassium, Request, Response

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

app = Potassium("my_app")

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
    model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
    model.half().to(torch.cuda.current_device())
   
    context = {
        "model": model,
        "tokenizer":tokenizer
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    model = context.get("model")
    tokenizer = context.get("tokenizer")
    
    prompt = request.json.get("prompt")
    max_new_tokens= request.json.get('max_new_tokens', 64)
    temperature= request.json.get('temperature', 0.7)

    system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
    - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
    - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
    - StableLM will refuse to participate in anything that could harm a human.
    """

    if prompt == None:
        return {'message': "No prompt provided"}
    
    full_prompt = system_prompt + prompt

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()])
        )

    return Response(
        json = {"output": tokenizer.decode(tokens[0], skip_special_tokens=True)}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()