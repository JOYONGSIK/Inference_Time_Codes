import transformers
import torch
import time

# Hugging Face 토큰 설정
import os 
os.environ["HUGGING_FACE_HUB_TOKEN"] = ""

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "lmsys/vicuna-7b-v1.5"


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You're a smart captioning model."},
    {"role": "user", "content": "Generate a single caption for an imaginary random image in 55 words."},
]

##### TIME START ! #####
start_time = time.time()
for _ in range(100):
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
end_time = time.time() ##### TIME STOP ! #####     
elapsed_time = end_time - start_time

print("Elapsed time: {:.6f}sec".format(elapsed_time))

# Elapsed time: 309.522255sec 
