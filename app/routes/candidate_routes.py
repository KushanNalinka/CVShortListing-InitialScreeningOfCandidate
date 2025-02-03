from flask import Blueprint, request, jsonify, send_from_directory
from app.models.candidate_model import Candidate
from app.cv_processing import (
    extract_text_from_pdf,
      extract_contact_info, 
      extract_soft_skills, 
      extract_technical_skills_from_projects, 
      generate_and_store_skills_chart,
      extract_programming_languages,
      extract_website_development_technologies,
      extract_programming_frameworks,
      extract_cloud_technologies,
      extract_devops_technologies,
      extract_version_control_technologies,
      extract_database_technologies,
      extract_project_experiences,
      extract_courses_certifications_achievements,
      extract_work_experience,
      extract_software_development_methodologies
)
import os
import json
from collections import Counter
from app.similarity import calculate_similarity
from app.utils.chart_utils import get_and_display_chart
from app.models.job_model import Job
import joblib
import pandas as pd

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
        no_devops_tools = extract_devops_technologies(cv_text)
        no_cloud_technologies = extract_cloud_technologies(cv_text)
        no_programming_frameworks = extract_programming_frameworks(cv_text)
        no_web_technologies = extract_website_development_technologies(cv_text)
        no_programming_languages = extract_programming_languages(cv_text)
        no_version_control_technologies = extract_version_control_technologies(cv_text)
        no_database_technologies = extract_database_technologies(cv_text)
        no_software_development_methodologies = extract_software_development_methodologies(cv_text)

        project_experiences = extract_project_experiences(cv_text)
        courses_certifications_achievements = extract_courses_certifications_achievements(cv_text)
        work_experience = extract_work_experience(cv_text)


        # Add processed data to the candidate data
        data.update({
            'extracted_email': contact_info.get('email'),
            'extractedgithub': contact_info.get('github'),
            'extractedlinkedin': contact_info.get('linkedin'),
            'extractednoofprogramminglanguages': no_programming_languages,
            'extractednoofwebtechnologies': no_web_technologies,
            'extractednoofprogrammingframeworks': no_programming_frameworks,
            'extractednoofcloudtechnologies': no_cloud_technologies,
            'extractednoofdevopstools': no_devops_tools,
            'extractednoofversioncontroltechnologies': no_version_control_technologies,
            'extractednoofdatabasetechnologies': no_database_technologies,
            'project_experiences': project_experiences,
            'extractednoofsoftwaredevelopmentmethodologies': no_software_development_methodologies,

            'courses_certifications_achievements': courses_certifications_achievements,
            'work_experience': work_experience,


        
            'cv_text': cv_text
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

    # Trigger post-save function to process CV further
    process_similarity_candidate_data(candidate['_id'])

    return jsonify(candidate), 201
# Load the model
model_path = os.path.join(os.getcwd(), "local_model", "stacked_model.joblib")
model = joblib.load(model_path)

feature_columns_path = os.path.join(os.getcwd(), "local_model", "feature_columns.pkl")
feature_columns = joblib.load(feature_columns_path)

def process_similarity_candidate_data(candidate_id):
    # Fetch the candidate data from the database
    candidate = Candidate.get_by_id(candidate_id)
    if not candidate:
        return jsonify({"error": "Candidate not found"}), 404

    # Extract and prepare candidate data
    courses_certifications_achievements = candidate.get("courses_certifications_achievements", [])
    work_experience = candidate.get("work_experience", [])
    project_experiences = candidate.get("project_experiences", [])

    # Ensure the candidate data is properly formatted into strings
    courses_certifications_text = " ".join(item.get("content", "") if isinstance(item, dict) else item for item in courses_certifications_achievements)
    work_experience_text = " ".join(work_experience)
    projects_text = " ".join(project_experiences)

    if not (courses_certifications_text.strip() or work_experience_text.strip() or projects_text.strip()):
        return jsonify({"error": "Candidate data missing for similarity calculation"}), 400
    

# Initialize all expected feature columns to 0
    candidate_data = {col: 0 for col in feature_columns}

    candidate_data = {
       "Years of Experience": candidate.get("noofyearsofexperience", 0),
        "Roles Marks": candidate.get("roles_marks", 0),
        "Programming Languages": candidate.get("extractednoofprogramminglanguages", 0),
        "Programming Frameworks": candidate.get("extractednoofprogrammingframeworks", 0),
        "Software Dev Methods": candidate.get("extractednoofsoftwaredevelopmentmethodologies", 0),
        "Web Dev Technologies": candidate.get("extractednoofwebtechnologies", 0),
        "Database Technologies": candidate.get("extractednoofdatabasetechnologies", 0),
        "Version Control": candidate.get("extractednoofversioncontroltechnologies", 0),
        "Courses & Certifications Matching %": candidate.get("coursesAndCertificationMatchingSimilarity", 0),
        "Achievements Matching %": candidate.get("achievements_similarity", 0),
        "Work Experience Matching %": candidate.get("workExperienceMatchingSimilarity", 0),
        "Projects Matching %": candidate.get("projectsMatchingSimilarity", 0),
        "Cloud Technologies": candidate.get("extractednoofcloudtechnologies", 0),
        "CI/CD Technologies": candidate.get("extractednoofdevopstools", 0),
        f"Position_{candidate.get('jobPosition', '')}": 1,
        f"Education Level_{candidate.get('education_level', '')}": 1,
        f"Specialization_{candidate.get('specialization', '')}": 1,
        f"University_{candidate.get('university', '')}": 1
    }
    
    # Convert to DataFrame and reindex to match the model's feature columns
    input_df = pd.DataFrame([candidate_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Predict matching percentage
    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Extract job details using jobID
    job_id = candidate.get("jobID")
    if not job_id:
        return jsonify({"error": "Job ID not associated with candidate"}), 400

    job = Job.get_by_id(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    # Fetch and validate job details
    qualifications = job.get("qualifications", "")
    required_skills = job.get("requiredSkills", "")
    duties = job.get("duties", "")
    experience = job.get("experience", "")

    if not (qualifications.strip() and required_skills.strip() and duties.strip() and experience.strip()):
        return jsonify({"error": "Job details incomplete for similarity calculation"}), 400

    # Perform similarity calculations
    courses_and_certifications_similarity = calculate_similarity(courses_certifications_text, qualifications)
    work_experience_similarity = calculate_similarity(work_experience_text, experience)
    projects_similarity = calculate_similarity(projects_text, required_skills)

    # Update the candidate record in the database
    update_data = {
        "coursesAndCertificationMatchingSimilarity": float(courses_and_certifications_similarity),
        "workExperienceMatchingSimilarity": float(work_experience_similarity),
        "projectsMatchingSimilarity": float(projects_similarity),
        "predicted_matching_percentage": round(prediction, 2)
    }
    Candidate.update(candidate_id, update_data)

    return jsonify({"message": "Similarity scores updated successfully"}), 200




def process_candidate_data(candidate_id):
    # Fetch the candidate data from the database
    candidate = Candidate.get_by_id(candidate_id)
    if not candidate:
        return

    cv_text = candidate.get('cv_text', '')
    if not cv_text:
        return

    # Perform soft skills extraction and similarity calculation
    soft_skills = extract_soft_skills(cv_text)
    soft_skills_text = " ".join(skill[0] for skill in soft_skills)
    similarity_score = calculate_similarity(soft_skills_text, cv_text)

    # Perform technical skills extraction
    technical_skills = extract_technical_skills_from_projects(cv_text)
    skill_counts = Counter(technical_skills)

    # Generate and store the technical skills bar chart as an image
    chart_image_base64 = generate_and_store_skills_chart(skill_counts)

    # Update the candidate record in the database
    update_data = {
        'soft_skills': [{"skill": skill[0], "count": skill[1]} for skill in soft_skills],
        'similarity_score': float(similarity_score),
         'technical_skills': [{"skill": skill, "count": count} for skill, count in skill_counts.items()],
        'skills_chart_image': chart_image_base64  # Store the base64-encoded image
    }
    Candidate.update(candidate_id, update_data)
    
@candidate_routes.route('/candidates/<candidateID>/process', methods=['POST'])
def process_candidate(candidateID):
    process_candidate_data(candidateID)
    return jsonify({"message": "Candidate processing triggered successfully"}), 200


    

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

@candidate_routes.route('/candidates/<candidateID>/chart', methods=['GET'])
def display_candidate_chart(candidateID):
    get_and_display_chart(candidateID)
    return jsonify({"message": "Chart displayed successfully"}), 200

