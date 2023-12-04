from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor

class ImageCaptioningDataset(Dataset):

    def __init__(self, processor):
        self.processor = processor
        self.value = torch.randint(100, (1, 3, 80, 512))

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        encoding = self.processor(images=self.value, text='discharge notes go here', padding="max_length", return_tensors="pt")

        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding
    
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

train_dataset = ImageCaptioningDataset(processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()

for epoch in range(5):
  print("Epoch:", epoch)
  for idx, batch in enumerate(train_dataloader):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)

    outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)
    
    loss = outputs.loss

    out = model.generate(pixel_values=train_dataset.value.to(device), max_length=50)
    print("Loss:", loss.item())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()



out = model.generate(pixel_values=train_dataset.value.to(device), max_length=50)
generated_caption = processor.batch_decode(out, skip_special_tokens=True)[0]
print(generated_caption)