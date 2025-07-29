from transformers import AutoTokenizer, OPTModel
import torch
import time

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
# model = OPTModel.from_pretrained("facebook/opt-125m")
model = OPTModel.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16, device_map="auto")  # 모델을 bfloat16으로 로드

inputs = tokenizer("You're a smart captioning model. Generate a single caption for an imaginary random image in 55 words.", 
                   return_tensors="pt").to("cuda")  
##### TIME START ! #####
start_time = time.time()
for _ in range(100):
    outputs = model(**inputs)
end_time = time.time() ##### TIME STOP ! #####     
elapsed_time = end_time - start_time

print("Elapsed time: {:.6f}sec".format(elapsed_time))

last_hidden_states = outputs.last_hidden_state