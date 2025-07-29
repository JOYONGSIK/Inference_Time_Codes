from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel
import time
import torch 


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
end_time = time.time() ##### TIME STOP ! #####     
elapsed_time = end_time - start_time

print("Elapsed time: {:.6f}sec".format(elapsed_time))


last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output 