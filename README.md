# Ideology Detection with BERT

## Project Overview
This project focuses on classifying text into five distinct ideologies using a fine-tuned BERT model. The dataset comprises ideological statements in Turkish, and the model is trained to distinguish between various ideological perspectives.

## Dataset
The dataset is sourced from ideological statements and consists of five categories:
- Anarchism
- Communism
- Liberalism
- Nationalism
- Conservatism

The dataset is stored in CSV format and is preprocessed to balance the number of samples for each ideology. The preprocessing steps include:
- Reading and cleaning the data
- Handling missing values
- Encoding labels
- Sampling to ensure class balance

## Model
The model is based on `dbmdz/bert-base-turkish-128k-uncased`, a pre-trained BERT model optimized for Turkish. It is fine-tuned for sequence classification using the Hugging Face `transformers` library. The model takes ideological text as input and predicts one of the five ideological categories.

## Environment Setup
### Requirements
Ensure you have the following dependencies installed:
```bash
pip install transformers torch pandas scikit-learn matplotlib seaborn
```

### GPU Check
The model is trained using CUDA. Ensure a GPU is available:
```python
import torch
if torch.cuda.is_available():
    print("GPU available: ", torch.cuda.get_device_name(0))
else:
    print("GPU not available")
```

## Data Preparation
The dataset is split into training (80%) and test (20%) sets. Text is tokenized using the BERT tokenizer, and attention masks are created to handle padding.

## Training
The model is trained for four epochs using the AdamW optimizer and a linear learning rate scheduler. Training loss is monitored at each step, and performance is evaluated using:
- Precision
- Recall
- F1-score

## Evaluation
The trained model is evaluated on the test set. The classification report provides metrics for each ideological category.

Example output:
```
F-Score:  0.7589
Recall:  0.7499
Precision:  0.7758
```

## Usage
To make predictions on new text samples:
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-uncased')
model = BertForSequenceClassification.from_pretrained("path/to/saved/model")
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

text_sample = "Devlet her zaman güçlülerin çıkarlarını korur."
print("Predicted Ideology: ", predict(text_sample))
```

## Model Checkpoint
The trained model is saved and can be loaded for inference.

## Authors
This project was developed by leveraging the Hugging Face `transformers` library and PyTorch for fine-tuning BERT on ideological classification tasks.

## License
This project is open-source and licensed under the MIT License.

