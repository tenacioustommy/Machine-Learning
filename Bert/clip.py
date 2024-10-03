from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image=Image.open("Bert/data/xiaoji.jpg")

text = ["the girl wants me more",  "i want the girl more"]
inputs=processor(text=text, images=image, return_tensors="pt",padding=True) 

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

for i in range(len(text)):
    print(f"{text[i]}: {probs[0][i]}")
