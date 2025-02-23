from modeling import *
import matplotlib.pyplot as plt

device = get_device()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b", 
    revision='float16', torch_dtype=torch.float16).to(device)
model = model.float()
wrapper = GPTJWrapper(model, tokenizer)


def tokenize(text):
    inp_ids = wrapper.tokenize(text)
    str_toks = wrapper.list_decode(inp_ids[0])
    return inp_ids, str_toks

poland_text="""Q: What is the capital of France?
A: Paris
Q: What is the capital of Poland?
A:"""

poland_ids, pol_toks = tokenize(poland_text)

wrapper.add_hooks()
out = wrapper.model(input_ids = poland_ids, output_hidden_states=True) #run it again to activate hooks
print('done')