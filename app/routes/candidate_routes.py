from flask import Blueprint, request, jsonify, send_from_directory
from app.models.candidate_model import Candidate
import os
import json
import joblib
import pandas as pd

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
      extract_software_development_methodologies,
      generate_all_charts,
      extract_achievements
)

from collections import Counter
from app.similarity import calculate_similarity
from app.utils.chart_utils import get_and_display_chart

from app.models.job_model import Job


candidate_routes = Blueprint('candidates', __name__)

from app import db  # Import the shared db instance
candidates_collection = db["candidates"]  # Define candidates_collection

# Define folders for CV and transcripts
UPLOAD_FOLDER_CV = os.path.join(os.getcwd(), 'uploads/cv')
UPLOAD_FOLDER_TRANSCRIPTS = os.path.join(os.getcwd(), 'uploads/transcripts')

os.makedirs(UPLOAD_FOLDER_CV, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_TRANSCRIPTS, exist_ok=True)

# Load the trained stacked model
model_path = os.path.join(os.getcwd(), "local_model", "stacked_model_new.joblib")
model = joblib.load(model_path)

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
        
        # Extract tools and technologies count
        no_devops_tools = extract_devops_technologies(cv_text)
        no_cloud_technologies = extract_cloud_technologies(cv_text)
        no_programming_frameworks = extract_programming_frameworks(cv_text)
        no_web_technologies = extract_website_development_technologies(cv_text)
        no_programming_languages = extract_programming_languages(cv_text)
        no_version_control_technologies = extract_version_control_technologies(cv_text)
        no_database_technologies = extract_database_technologies(cv_text)
        no_software_development_methodologies = extract_software_development_methodologies(cv_text)

        # Extract candidate experience details
        project_experiences = extract_project_experiences(cv_text)
        courses_certifications_achievements = extract_courses_certifications_achievements(cv_text)
        work_experience = extract_work_experience(cv_text)
        achievements = extract_achievements(cv_text)

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
            'extractednoofsoftwaredevelopmentmethodologies': no_software_development_methodologies,
            'project_experiences': project_experiences,
            'courses_certifications_achievements': courses_certifications_achievements,
            'achievements': achievements,
            'work_experience': work_experience,
            'cv_text': cv_text
        })

    # Save the uploaded transcript file
    if transcript:
        transcript_path = os.path.join(UPLOAD_FOLDER_TRANSCRIPTS, transcript.filename)
        transcript.save(transcript_path)
        data['transcript'] = transcript.filename

    # Save the structured data to the database
    candidate = Candidate.create(data)

    # Trigger similarity and prediction processing
    process_similarity_and_prediction(candidate['_id'])

    return jsonify(candidate), 201

def process_similarity_and_prediction(candidate_id):
    candidate = Candidate.get_by_id(candidate_id)
    if not candidate:
        return jsonify({"error": "Candidate not found"}), 404

    # Extract textual data for similarity calculation
    courses_certifications_text = " ".join(
        item.get("content", "") if isinstance(item, dict) else item
        for item in candidate.get("courses_certifications_achievements", [])
    )
    work_experience_text = " ".join(candidate.get("work_experience", []))
    projects_text = " ".join(candidate.get("project_experiences", []))
    achievements_text = " ".join(candidate.get("achievements", []))

    # Handle missing job data
    job_id = candidate.get("jobID")
    if not job_id:
        return jsonify({"error": "Job ID missing"}), 400

    job = Job.get_by_id(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    required_skills = job.get("skills", "")
    experience = job.get("experience", "")
    qualifications = job.get("qualifications", "")
    achievements = job.get("duties", "")

    if not any([qualifications.strip(), required_skills.strip(), experience.strip(), achievements.strip()]):
        return jsonify({"error": "Job details missing"}), 400

    
    # Calculate similarity scores (converted to percentages)
    courses_similarity = round(calculate_similarity(courses_certifications_text, qualifications) * 100, 2)
    work_experience_similarity = round(calculate_similarity(work_experience_text, experience) * 100, 2)
    projects_similarity = round(calculate_similarity(projects_text, required_skills) * 100, 2)
    achievements_similarity = round(calculate_similarity(achievements_text, achievements) * 100, 2)

    # Count number of tools & technologies
    num_of_tools_technologies = sum([
        candidate.get("extractednoofprogramminglanguages", 0),
        candidate.get("extractednoofwebtechnologies", 0),
        candidate.get("extractednoofprogrammingframeworks", 0),
        candidate.get("extractednoofcloudtechnologies", 0),
        candidate.get("extractednoofdevopstools", 0),
        candidate.get("extractednoofversioncontroltechnologies", 0),
        candidate.get("extractednoofdatabasetechnologies", 0),
        candidate.get("extractednoofsoftwaredevelopmentmethodologies", 0)
    ])

    # Update candidate record with similarity scores
    Candidate.update(candidate_id, {
        "coursesAndCertificationMatchingSimilarity": courses_similarity,
        "workExperienceMatchingSimilarity": work_experience_similarity,
        "projectsMatchingSimilarity": projects_similarity,
        "achievements_similarity": achievements_similarity,
        "num_of_tools_technologies": num_of_tools_technologies
    })

    return jsonify({"message": "Candidate similarity updated"}), 200

@candidate_routes.route('/candidates/predict/<candidate_id>', methods=['GET'])
def predict_matching_percentage(candidate_id):
    """Predicts the candidate's matching percentage using an ML model."""
    candidate = Candidate.get_by_id(candidate_id)
    if not candidate:
        return jsonify({"error": "Candidate not found"}), 404

    input_data = {
        "No of Tools and Technologies": candidate.get("num_of_tools_technologies", 0),
        "Courses & Certifications Matching": candidate.get("coursesAndCertificationMatchingSimilarity", 0),
        "Achievements Matching": candidate.get("achievements_similarity",0),  # Assuming static value for achievements
        "Work Experience Matching": candidate.get("workExperienceMatchingSimilarity", 0),
        "Projects Matching": candidate.get("projectsMatchingSimilarity", 0)
    }

    input_df = pd.DataFrame([input_data])

    try:
        predicted_matching_percentage = model.predict(input_df)[0]
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    # Update candidate record with prediction
    Candidate.update(candidate_id, {
        "predicted_matching_percentage": round(predicted_matching_percentage, 2)
    })

    return jsonify({
        "predicted_matching_percentage": round(predicted_matching_percentage, 2)
    }), 200

# @candidate_routes.route('/candidates/predict/<candidate_id>', methods=['POST'])
# def predict_matching_percentage(candidate_id):
#     candidate = candidates_collection.find_one({"_id": candidate_id})
#     if not candidate:
#         return jsonify({"error": "Candidate not found"}), 404
#     input_data = {
#         "No of Tools and Technologies": candidate.get("num_of_tools_technologies", 0),
#         "Courses & Certifications Matching": candidate.get("coursesAndCertificationMatchingSimilarity", 0),
#         "Achievements Matching": candidate.get("achievements_similarity", 0),
#         "Work Experience Matching": candidate.get("workExperienceMatchingSimilarity", 0),
#         "Projects Matching": candidate.get("projectsMatchingSimilarity", 0)
#     }
#     input_df = pd.DataFrame([input_data])
#     try:
#         predicted_matching_percentage = model.predict(input_df)[0]
#         candidates_collection.update_one(
#             {"_id": candidate_id},
#             {"$set": {"predicted_matching_percentage": round(predicted_matching_percentage, 2)}}
#         )
#         return jsonify({"predicted_matching_percentage": round(predicted_matching_percentage, 2)}), 200
#     except Exception as e:
#         return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@candidate_routes.route('/candidates/predicted_percentage/<candidate_id>', methods=['GET'])
def get_predicted_percentage(candidate_id):
    candidate = candidates_collection.find_one({"_id": candidate_id}, {"predicted_matching_percentage": 1})
    if not candidate:
        return jsonify({"error": "Candidate not found"}), 404
    return jsonify({"predicted_matching_percentage": candidate.get("predicted_matching_percentage", "N/A")}), 200




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
    
@candidate_routes.route('/candidates/candidatesByGeneratedId/<generatedCandidateID>', methods=['GET'])
def get_candidate_by_generated_id(generatedCandidateID):
    
        # Find the candidate using the generated candidate ID
        genCandidate = candidates_collection.find_one({"candidateId": generatedCandidateID})

        if not genCandidate:
            return jsonify({"error": "Candidate not found"}), 404

        # Extract the MongoDB ObjectId
        candidateID = genCandidate.get('_id')

        # Fetch the full candidate details using the ObjectId
        candidate = Candidate.get_by_id(candidateID)

        if candidate:
          
            return jsonify(candidate), 200
        else:
            return jsonify({"error": "Candidate not found"}), 404
        
@candidate_routes.route('/candidates/job/<jobId>', methods=['GET'])
def get_candidates_by_job(jobId):
    try:
        # Find candidates using jobId (make sure jobId is stored as a string in MongoDB)
        candidates = list(candidates_collection.find({"jobID": jobId}))

        if not candidates:
            return jsonify({"error": "No candidates found for this job"}), 404

        # Convert ObjectId to string before returning the response
        for candidate in candidates:
            candidate['_id'] = str(candidate['_id'])  # Convert ObjectId to string

        return jsonify(candidates), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


  

        
      

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

@candidate_routes.route('/candidates/generate_charts/<candidateID>', methods=['POST'])
def create_all_charts(candidateID):
    candidate = Candidate.get_by_id(candidateID)
    if not candidate:
        return jsonify({"error": "Candidate not found"}), 404

    cv_text = candidate.get('cv_text', '')
    if not cv_text:
        return jsonify({"error": "CV text not available"}), 400

    charts = generate_all_charts(cv_text)
    Candidate.update(candidateID, {"charts": charts})
    
    return jsonify({"message": "Charts generated successfully", "charts": charts}), 200

@candidate_routes.route('/candidates/charts/<candidateID>', methods=['GET'])
def get_all_charts(candidateID):
    candidate = Candidate.get_by_id(candidateID)
    if not candidate:
        return jsonify({"error": "Candidate not found"}), 404

    charts = candidate.get('charts', {})
    return jsonify({"charts": charts}), 200


