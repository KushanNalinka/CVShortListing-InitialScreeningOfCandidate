import PyPDF2
import spacy
import re
from collections import Counter
from transformers import pipeline
import matplotlib.pyplot as plt
import base64
from io import BytesIO


nlp = spacy.load("en_core_web_sm")


classifier = pipeline("zero-shot-classification", model="local_model", tokenizer="local_model")


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text


def extract_contact_info(text):
    email = re.findall(r'\S+@\S+', text)
    github = re.findall(r'https?://(?:www\.)?github\.com/[^\s]+', text)
    linkedin = re.findall(r'https?://(?:www\.)?linkedin\.com/[^\s]+', text)
    return {
        'email': email[0] if email else None,
        'github': github[0] if github else None,
        'linkedin': linkedin[0] if linkedin else None
    }


def extract_soft_skills(cv_text):
    soft_skills_list = [
        'communication', 'teamwork', 'leadership', 'problem-solving', 'adaptability',
        'time management', 'critical thinking', 'collaboration', 'creativity',
        'decision-making', 'empathy', 'negotiation', 'conflict resolution',
        'responsibility', 'accountability'
    ]
    soft_skills_found = []
    doc = nlp(cv_text)
    for sent in doc.sents:
        results = classifier(sent.text, candidate_labels=soft_skills_list, multi_label=True)
        for label, score in zip(results['labels'], results['scores']):
            if score >= 0.6 and label not in soft_skills_found:
                soft_skills_found.append(label)
    return Counter(soft_skills_found).most_common()

# Function to extract technical skills from project descriptions
def extract_technical_skills_from_projects(text):
    doc = nlp(text.lower())
    extracted_technical_skills = []
    for sentence in doc.sents:
        sentence_text = sentence.text.strip()
        for skill in all_technical_skills:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', sentence_text):
                extracted_technical_skills.append(skill)
    return extracted_technical_skills

# Function to generate and return the base64-encoded skills bar chart
def generate_and_store_skills_chart(skill_counts):
    # Extract skill names and their counts
    skills = list(skill_counts.keys())
    counts = list(skill_counts.values())

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(skills, counts, color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Technical Skills')
    plt.title('Technical Skills Extracted from Resume')
    plt.tight_layout()

    # Save the chart to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Convert the image to base64
    chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return chart_base64



