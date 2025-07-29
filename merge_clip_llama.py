import transformers
import torch
import time

from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel

# Hugging Face 토큰 설정
import os 
os.environ["HUGGING_FACE_HUB_TOKEN"] = ""

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

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
    llama_outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    outputs = model(**inputs)
end_time = time.time() ##### TIME STOP ! #####     
elapsed_time = end_time - start_time

print("Elapsed time: {:.6f}sec".format(elapsed_time))