from predictor import model, processor, device

from PIL import Image

import torch



image_path = "uploads/real_00010.jpg"



image = Image.open(
    image_path
).convert("RGB")



inputs = processor(
    images=image,
    return_tensors="pt"
)



inputs = {
    k:v.to(device)
    for k,v in inputs.items()
}



with torch.no_grad():

    output = model(
        **inputs
    )



prob = torch.softmax(
    output.logits,
    dim=1
)



print("Real probability :", 
      prob[0][0].item()*100)


print("Fake probability :", 
      prob[0][1].item()*100)