from flask import Blueprint, request, jsonify, send_from_directory
from app.models.candidate_model import Candidate
from app.cv_processing import extract_text_from_pdf, extract_contact_info, extract_soft_skills
import os
import json
from app.similarity import calculate_similarity

candidate_routes = Blueprint('candidates', __name__)

# Define folders for CV and transcripts
UPLOAD_FOLDER_CV = os.path.join(os.getcwd(), 'uploads/cv')
UPLOAD_FOLDER_TRANSCRIPTS = os.path.join(os.getcwd(), 'uploads/transcripts')

os.makedirs(UPLOAD_FOLDER_CV, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_TRANSCRIPTS, exist_ok=True)

@candidate_routes.route('/candidates', methods=['POST'])
def create_candidate():
    data = request.form.to_dict()  # Parse form data
    cv = request.files.get('resume')
    transcript = request.files.get('transcript')

    # Save the uploaded CV file
    if cv:
        cv_path = os.path.join(UPLOAD_FOLDER_CV, cv.filename)
        cv.save(cv_path)
        data['resume'] = cv.filename

        # Extract and process CV data
        cv_text = extract_text_from_pdf(cv_path)
        contact_info = extract_contact_info(cv_text)
        soft_skills = extract_soft_skills(cv_text)

        # Calculate similarity score
        soft_skills_text = " ".join(skill[0] for skill in soft_skills)
        similarity_score = calculate_similarity(soft_skills_text, cv_text)

        # Add processed data to the candidate data
        data.update({
            'extracted_email': contact_info.get('email'),
            'extractedgithub': contact_info.get('github'),
            'extractedlinkedin': contact_info.get('linkedin'),
            'soft_skills': [{"skill": skill[0], "count": skill[1]} for skill in soft_skills],
            'similarity_score': float(similarity_score)  # Ensure Python float
        })

    # Save the uploaded transcript file
    if transcript:
        transcript_path = os.path.join(UPLOAD_FOLDER_TRANSCRIPTS, transcript.filename)
        transcript.save(transcript_path)
        data['transcript'] = transcript.filename

    # Parse JSON strings into Python objects
    data['experience'] = json.loads(data.get('experience', '[]'))
    data['education'] = json.loads(data.get('education', '[]'))

    # Save the structured data to the database
    candidate = Candidate.create(data)
    return jsonify(candidate), 201

@candidate_routes.route('/candidates', methods=['GET'])
def get_candidates():
    candidates = Candidate.get_all()
    return jsonify(candidates), 200

@candidate_routes.route('/candidates/<candidateID>', methods=['DELETE'])
def delete_candidate(candidateID):
    Candidate.delete(candidateID)
    return jsonify({"message": "Candidate deleted"}), 200

@candidate_routes.route('/candidates/<candidateID>', methods=['GET'])
def get_candidate(candidateID):
    candidate = Candidate.get_by_id(candidateID)
    if candidate:
        return jsonify(candidate), 200
    else:
        return jsonify({"error": "Candidate not found"}), 404

@candidate_routes.route('/uploads/cv/<filename>', methods=['GET'])
def serve_cv(filename):
    return send_from_directory(UPLOAD_FOLDER_CV, filename)

@candidate_routes.route('/uploads/transcripts/<filename>', methods=['GET'])
def serve_transcript(filename):
    return send_from_directory(UPLOAD_FOLDER_TRANSCRIPTS, filename)

