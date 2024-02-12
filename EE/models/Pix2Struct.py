import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration


def collator(batch, processor):
    new_batch = {"flattened_patches": [], "attention_mask": []}
    texts = [item["text"] for item in batch]

    text_inputs = processor(
        text=texts,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=True,
        max_length=20,
    )

    new_batch["labels"] = text_inputs.input_ids

    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch


class Pix2Struct(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.processor = AutoProcessor.from_pretrained(config["model_weights"])
        self.model = Pix2StructForConditionalGeneration.from_pretrained(
            config["model_weights"]
        )
        self.batch_size = config["batch_size"]
        self.device = config["device"]

    def parallelize(self):
        self.model = nn.DataParallel(self.model)

    def forward(self, batch, return_confidence=False):
        # for k, v in batch.items():
        #     print(k, v.shape)
        outputs = self.model(**batch)

        return outputs


def test_processor():
    import requests

    processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")

    # labels = "Blabla"
    # image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)  # dummy image
    labels = "A random invoice from Car Doctors with GBP currency and a total of 336.00"
    url = "https://images.ctfassets.net/txhaodyqr481/4fC2cAwG0mSmTtl58mxG8n/ee5331316aa95011c9394273f5c35133/New_Project__9_.jpg?fm=jpg&q=85&fl=progressive&w=1200&h=1696"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(
        images=image, text=labels, return_tensors="pt", add_special_tokens=True
    )

    print(inputs)
    model = Pix2StructForConditionalGeneration.from_pretrained(
        "google/pix2struct-textcaps-base"
    )
    from pdb import set_trace

    set_trace()
    # outputs = model(**inputs)
    # last_hidden_states = outputs.loss

    flattened_patches = inputs.flattened_patches
    attention_mask = inputs.attention_mask

    generated_ids = model.generate(
        flattened_patches=flattened_patches,
        attention_mask=attention_mask,
        max_length=50,
    )
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ]
    print(generated_caption)


if __name__ == "__main__":
    test_processor()
