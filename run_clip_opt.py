from transformers import AutoTokenizer, OPTModel

from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel

import torch
import time 

# OPT 
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
opt_model = OPTModel.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16, device_map="cuda")  # 모델을 bfloat16으로 로드
opt_inputs = tokenizer("You're a smart captioning model. Generate a single caption for an imaginary random image in 55 words.", 
                   return_tensors="pt").to("cuda")  

# CLIP 

model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336",torch_dtype=torch.float16).cuda() # 모델 float16으로 로드. 
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt").to("cuda")
inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16) # 여기 코드 추가. 

##### TIME START ! #####
start_time = time.time()
for _ in range(100):
    outputs = model(**inputs)
    opt_outputs = opt_model(**opt_inputs)
end_time = time.time() ##### TIME STOP ! ##### 
    
elapsed_time = end_time - start_time

print("Elapsed time: {:.6f}sec".format(elapsed_time))