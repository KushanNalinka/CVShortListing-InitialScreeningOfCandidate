from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

def calculate_similarity(paragraph1, paragraph2):
    # Tokenize paragraphs
    tokens = {'input_ids': [], 'attention_mask': []}
    for paragraph in [paragraph1, paragraph2]:
        new_tokens = tokenizer.encode_plus(
            paragraph,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # Stack tokens into tensors
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

    # Get model outputs
    with torch.no_grad():
        outputs = model(**tokens)

    # Apply mean pooling for embeddings
    attention_mask = tokens['attention_mask'].unsqueeze(-1).float()
    embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1) / attention_mask.sum(1)

    # Convert embeddings to NumPy
    embeddings = embeddings.detach().numpy()

    # Calculate cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)  # Ensure it returns a standard Python float
