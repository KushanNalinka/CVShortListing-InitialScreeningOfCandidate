from transformers import pipeline

# Initialize the pipeline with the pre-trained model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Save the model to the 'local_model' folder
classifier.model.save_pretrained("local_model")
classifier.tokenizer.save_pretrained("local_model")

print("Model has been saved to the 'local_model' folder.")
