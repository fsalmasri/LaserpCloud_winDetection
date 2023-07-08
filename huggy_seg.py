from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from transformers import pipeline
from transformers import AutoImageProcessor


checkpoint = "nvidia/mit-b0"
image_processor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available, otherwise use a CPU
# encoding = image_processor(image, return_tensors="pt")
# pixel_values = encoding.pixel_values.to(device)