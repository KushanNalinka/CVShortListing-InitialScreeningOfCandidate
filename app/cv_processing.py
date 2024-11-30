import PyPDF2
import spacy
import re
from collections import Counter
from transformers import pipeline



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



