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

# Define dictionaries for technical skills
skills_keywords = {
    'Programming Languages': [
        'Python', 'Java', 'C', 'C++', 'C#', 'JavaScript', 'TypeScript', 'PHP', 'Ruby', 'Go', 'Swift', 'Kotlin', 'Rust'
    ],
    'Frameworks and Libraries': [
        'React', 'Angular', 'Node.js', 'Django', 'Flask', 'Spring', 'Laravel', 'Vue.js', 'Bootstrap', 'TensorFlow',
        'Keras', 'PyTorch', 'jQuery', 'Express', 'Next.js', 'Spring Boot', 'Material UI', 'Hibernate', 'FastAPI'
    ],
    'Databases': [
        'MySQL', 'MongoDB', 'PostgreSQL', 'SQLite', 'Oracle', 'SQL Server', 'Firebase', 'DynamoDB', 'Redis'
    ],
    'Cloud Platforms': [
        'AWS', 'Azure', 'Google Cloud', 'Firebase', 'Heroku', 'DigitalOcean', 'Cloudflare', 'IBM Cloud'
    ],
    'Tools and Technologies': [
        'Docker', 'Kubernetes', 'Git', 'GitHub', 'GitLab', 'JIRA', 'Bitbucket', 'Jenkins', 'Ansible', 'Terraform',
        'CI/CD', 'Serverless', 'Kibana', 'Elasticsearch', 'Logstash', 'Splunk', 'Figma', 'Postman'
    ],
    'Development Methodologies': [
        'Agile', 'Scrum', 'Kanban', 'DevOps', 'Waterfall', 'Test-Driven Development', 'Behavior-Driven Development'
    ]
}

# Combine all technical skills into a single list for easier matching
all_technical_skills = [skill for category in skills_keywords.values() for skill in category]

    

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

def extract_programming_languages(text):
    programming_languages = [
        'Python', 'Java', 'C', 'C\\+\\+', 'C#', 'JavaScript', 'Ruby', 'Go', 'Swift', 'Kotlin', 'PHP',
        'TypeScript', 'R', 'Perl', 'Objective-C', 'Rust', 'Scala', 'Dart', 'Haskell', 'MATLAB',
        'Shell', 'PowerShell', 'SQL', 'Bash', 'HTML', 'CSS', 'SASS', 'Fortran', 'COBOL'
    ]
    
    found_languages = []

    programming_languages_escaped = [re.escape(lang) for lang in programming_languages]
    pattern = re.compile(r'\b(' + '|'.join(programming_languages_escaped) + r')\b', re.IGNORECASE)
    matches = pattern.findall(text)
    found_languages.extend(set(matches))

    doc = nlp(text)
    for sent in doc.sents:
        sentence_text = sent.text.lower()
        for lang in programming_languages:
            if lang.lower() in sentence_text and lang not in found_languages:
                found_languages.append(lang)

    return len(found_languages)

def extract_website_development_technologies(text):
    web_technologies = [
        'HTML', 'CSS', 'JavaScript', 'TypeScript', 'PHP', 'Ruby', 'Python', 'Java', 'C#', 
        'React', 'Angular', 'Vue.js', 'Svelte', 'Bootstrap', 'Tailwind CSS', 'jQuery', 
        'Node.js', 'Express', 'Next.js', 'Nuxt.js', 'Django', 'Flask', 'Laravel', 'ASP.NET',
        'Ruby on Rails', 'Spring', 'WordPress', 'Magento', 'Shopify', 'Joomla', 'Drupal',
        'Firebase', 'MongoDB', 'MySQL', 'PostgreSQL', 'SQLite', 'Redis', 'GraphQL', 'REST API',
        'WebSockets', 'Webpack', 'Gulp', 'Grunt', 'Parcel', 'Sass', 'LESS', 'JSON', 'AJAX',
        'WebAssembly', 'Three.js', 'Babel', 'Handlebars', 'Pug', 'Materialize', 'Foundation'
    ]
    
    found_web_technologies = []

    pattern = re.compile(r'\b(' + '|'.join(web_technologies) + r')\b', re.IGNORECASE)
    matches = pattern.findall(text)
    found_web_technologies.extend(set(matches))

    doc = nlp(text)
    for sent in doc.sents:
        sentence_text = sent.text.lower()
        if 'web development' in sentence_text or 'website' in sentence_text:
            for tech in web_technologies:
                if tech.lower() in sentence_text and tech not in found_web_technologies:
                    found_web_technologies.append(tech)

    return len(found_web_technologies)

def extract_programming_frameworks(text):
    frameworks = [
        'Django', 'Flask', 'React', 'Angular', 'Vue.js', 'Spring', 'Express', 'Rails', 'Ruby on Rails',
        'Laravel', 'ASP.NET', 'Node.js', 'Next.js', 'Bootstrap', 'Tailwind CSS', 'Foundation',
        'jQuery', 'Svelte', 'Nuxt.js', 'Redux', 'Keras', 'TensorFlow', 'Pandas', 'Hadoop',
        'Spark', 'Vue', 'Symfony', 'Zend', 'CakePHP', 'Meteor', 'FastAPI', 'Quasar', 'Phoenix', 'Ionic',
        'React Native', 'Electron', 'Flutter', 'Backbone.js', 'Alpine.js', 'NestJS', 'Struts', 'Sequelize',
        'CodeIgniter', 'Play Framework', 'Vaadin', 'Gatsby', 'Ember.js', 'MobX', 'CouchDB', 'RxJava'
    ]
    
    found_frameworks = []

    pattern = re.compile(r'\b(' + '|'.join(frameworks) + r')\b', re.IGNORECASE)
    matches = pattern.findall(text)
    found_frameworks.extend(set(matches))

    doc = nlp(text)
    for sent in doc.sents:
        sentence_text = sent.text.lower()
        for framework in frameworks:
            if framework.lower() in sentence_text and framework not in found_frameworks:
                found_frameworks.append(framework)

    return len(found_frameworks)

def extract_cloud_technologies(text):
    cloud_technologies = [
        'AWS', 'Amazon Web Services', 'Azure', 'Google Cloud', 'GCP', 'IBM Cloud', 'Oracle Cloud',
        'DigitalOcean', 'Alibaba Cloud', 'Salesforce', 'SAP Cloud', 'Cloud Foundry', 'Heroku', 
        'Red Hat OpenShift', 'VMware Cloud', 'Rackspace', 'Linode', 'Cloudflare', 'Backblaze',
        'Kubernetes', 'Docker', 'Terraform', 'Ansible', 'CloudFormation', 'Pulumi', 'Spinnaker', 
        'Vault', 'Consul', 'Istio', 'Anthos', 'Cloud Functions', 'Lambda', 'Azure Functions', 
        'Google Cloud Functions', 'Serverless', 'ECS', 'EKS', 'Fargate', 'AKS', 'OpenStack', 'VPC',
        'EC2', 'S3', 'RDS', 'BigQuery', 'Cloud Storage', 'Cloud Run', 'Elastic Beanstalk', 
        'CloudWatch', 'CloudTrail', 'CloudFront', 'IAM', 'Load Balancer', 'Auto Scaling', 'S3 Buckets'
    ]
    
    found_technologies = []

    pattern = re.compile(r'\b(' + '|'.join(cloud_technologies) + r')\b', re.IGNORECASE)
    matches = pattern.findall(text)
    found_technologies.extend(set(matches))

    doc = nlp(text)
    for sent in doc.sents:
        sentence_text = sent.text.lower()
        if 'cloud' in sentence_text or 'deployment' in sentence_text:
            for tech in cloud_technologies:
                if tech.lower() in sentence_text and tech not in found_technologies:
                    found_technologies.append(tech)

    return len(found_technologies)

def extract_devops_technologies(text):
    devops_tools = [
        'Jenkins', 'GitLab CI', 'GitHub Actions', 'Travis CI', 'CircleCI', 'TeamCity', 'Bamboo',
        'ArgoCD', 'Spinnaker', 'Flux', 'Harness', 'Azure DevOps', 'Octopus Deploy', 'CodePipeline',
        'Docker', 'Kubernetes', 'Ansible', 'Terraform', 'Chef', 'Puppet', 'Nagios', 'Prometheus',
        'Grafana', 'Elastic Stack', 'ELK Stack', 'Splunk', 'New Relic', 'AppDynamics', 'Sentry',
        'Consul', 'Vault', 'Istio', 'Linkerd', 'Nginx', 'Apache Kafka', 'AWS CodeDeploy', 'CloudFormation',
        'SaltStack', 'OpenShift', 'Vagrant'
    ]
    
    found_tools = []

    pattern = re.compile(r'\b(' + '|'.join(devops_tools) + r')\b', re.IGNORECASE)
    matches = pattern.findall(text)
    found_tools.extend(set(matches))

    doc = nlp(text)
    for sent in doc.sents:
        sentence_text = sent.text.lower()
        if 'ci/cd' in sentence_text or 'devops' in sentence_text:
            for tool in devops_tools:
                if tool.lower() in sentence_text and tool not in found_tools:
                    found_tools.append(tool)

    return len(found_tools)

def extract_version_control_technologies(cv_text):
    version_control_tools = [
        "Git", "SVN", "Subversion", "Mercurial", "Perforce", "Bazaar", 
        "GitHub", "GitLab", "Bitbucket", "Azure DevOps", "SourceForge",
        "TortoiseSVN", "SmartGit", "GitKraken", "SourceTree"
    ]
    
    vc_pattern = re.compile(r'\b(' + '|'.join(re.escape(tool) for tool in version_control_tools) + r')\b', re.IGNORECASE)
    found_tools = set(vc_pattern.findall(cv_text))

    doc = nlp(cv_text)
    verified_tools = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        for tool in found_tools:
            if tool.lower() in sentence_text.lower():
                verified_tools.append(tool)
    
    verified_tools = list(set(verified_tools))
    return len(verified_tools)

def extract_database_technologies(cv_text):
    database_technologies = [
        "MySQL", "PostgreSQL", "SQLite", "Oracle Database", "SQL Server", "MongoDB", "Cassandra",
        "DynamoDB", "CouchDB", "Redis", "Firebase", "Snowflake", "Redshift", "BigQuery",
        "Neo4j", "JanusGraph", "ArangoDB", "Elasticsearch", "MariaDB", "CockroachDB", 
        "HBase", "ClickHouse"
    ]
    
    db_pattern = re.compile(r'\b(' + '|'.join(re.escape(db) for db in database_technologies) + r')\b', re.IGNORECASE)
    found_databases = set(db_pattern.findall(cv_text))
    
    doc = nlp(cv_text)
    verified_databases = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        for db in found_databases:
            if db.lower() in sentence_text.lower():
                verified_databases.append(db)
    
    verified_databases = list(set(verified_databases))
    return len(verified_databases)


def extract_software_development_methodologies(cv_text):
    methodologies = [
        'Agile', 'Scrum', 'Kanban', 'Waterfall', 'Lean', 
        'Extreme Programming (XP)', 'DevOps', 'Spiral', 'RAD', 
        'V-Model', 'Incremental Development', 'Iterative Development', 
        'Prototyping'
    ]

    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(method) for method in methodologies) + r')\b', re.IGNORECASE)

    found_methodologies = pattern.findall(cv_text)

    return len(list(dict.fromkeys(found_methodologies)))


def extract_project_experiences(cv_text):
    projects = []
    project_keywords = ['project', 'technologies', 'tools', 'description', 'responsibilities']

    lines = cv_text.split('\n')
    current_project = None

    for line in lines:
        line_lower = line.lower().strip()
        if any(keyword in line_lower for keyword in project_keywords) and line:
            if current_project:
                projects.append(current_project.strip())
            current_project = line.strip()
        elif current_project:
            current_project += " " + line.strip()

    if current_project:
        projects.append(current_project.strip())

    filtered_projects = [project for project in projects if len(project.split()) > 5]

    return filtered_projects

def extract_courses_certifications_achievements(cv_text):
    sections = []
    
    course_keywords = ['course', 'certification', 'training', 'online course', 'workshop']
    achievement_keywords = ['achievement', 'award', 'recognition', 'honor', 'milestone']
    
    lines = cv_text.split('\n')
    current_section = None
    current_section_type = None

    for line in lines:
        line_lower = line.lower().strip()

        if any(keyword in line_lower for keyword in course_keywords):
            if current_section:
                sections.append({'type': current_section_type, 'content': current_section.strip()})
            current_section = line.strip()
            current_section_type = 'Courses/Certifications'
        elif any(keyword in line_lower for keyword in achievement_keywords):
            if current_section:
                sections.append({'type': current_section_type, 'content': current_section.strip()})
            current_section = line.strip()
            current_section_type = 'Achievements'
        elif current_section:
            current_section += " " + line.strip()

    if current_section:
        sections.append({'type': current_section_type, 'content': current_section.strip()})
    
    return sections

def extract_work_experience(cv_text):
    work_experience = []
    
    work_experience_keywords = ['work experience', 'employment history', 'professional experience', 'job', 'company', 'position', 'role']
    
    lines = cv_text.split('\n')
    current_experience = None

    for line in lines:
        line_lower = line.lower().strip()
        
        if any(keyword in line_lower for keyword in work_experience_keywords) and line:
            if current_experience:
                work_experience.append(current_experience.strip())
            current_experience = line.strip()
        elif current_experience:
            current_experience += " " + line.strip()

    if current_experience:
        work_experience.append(current_experience.strip())

    return work_experience
