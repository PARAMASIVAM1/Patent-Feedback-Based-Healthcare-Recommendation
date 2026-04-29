# Healthcare AI System

## Project Title
**NLP-Based Hospital, Specialist, and Doctor Recommendation System**

## Problem Statement
Patients often struggle to identify the most suitable hospital, specialist, or doctor because healthcare information is spread across reviews, ratings, consultation fees, waiting time, treatment quality, and service details. Existing systems mainly rely on basic ratings and do not effectively interpret patient feedback written in natural language.

This project proposes an NLP-based machine learning system that analyzes patient reviews, extracts sentiment and service-related features, and recommends hospitals, specialists, and doctors according to user preferences such as treatment quality, doctor behavior, consultation cost, waiting time, and specialization.

## Objective
- Analyze patient feedback written in natural language.
- Detect sentiment and complaint categories such as fees, waiting time, behavior, and treatment quality.
- Recommend the most suitable doctor or specialist using feedback similarity, ratings, experience, location, and fee constraints.
- Support multilingual feedback processing for a better user experience.

## Project Pipeline Diagram

User Input
→ Feedback / symptoms / preferences entered in the web form

Data Loading
→ Load doctor dataset, FAQ data, and patient review data

Preprocessing
→ Clean text, normalize fields, prepare labels, and filter by location, fee, and specialization

NLP Analysis
→ Traditional NLP: TF-IDF + Naive Bayes
→ Deep Learning NLP: SBERT / multilingual BERT embeddings
→ Sentiment and complaint analysis

Doctor Matching
→ Score doctors using ratings, semantic similarity, and experience

Recommendation Output
→ Show top doctors, hospitals, and guidance on the website

## Dataset Details

### 1. Main Doctor Dataset
- **File:** `dataset.csv`
- **Description:** Contains doctor profiles, consultation fees, locations, experience, ratings, feedback text, sentiment labels, and complaint categories.
- **Important columns:**
  - `doctor_id`
  - `doctor_name`
  - `specialization`
  - `department`
  - `experience_years`
  - `qualification`
  - `consultation_fee`
  - `location`
  - `average_rating`
  - `feedback_text`
  - `sentiment_label`
  - `complaint_category`

### 2. FAQ Dataset
- **File:** `faq.csv`
- **Description:** Contains healthcare question-answer pairs for common user queries and symptom guidance.
- **Important columns:**
  - `question`
  - `answer`
  - `keywords`

### 3. Processed Output Files
- `outputs/processed_tabular.csv`
- `phase2_outputs/processed_dataset.csv`
- `phase2_outputs/model_comparison.csv`
- `phase2_outputs/summary.json`

## Model Details

### Traditional Machine Learning Components
- **TF-IDF Vectorizer** for converting text into numerical features.
- **Multinomial Naive Bayes** for sentiment and feedback classification.
- **Cosine Similarity** for matching user feedback with doctor reviews.
- **Label Encoding** for categorical labels.

### Deep Learning / NLP Components
- **SBERT / Sentence Transformers** for semantic similarity.
- **Multilingual BERT embeddings** for understanding feedback in multiple languages.
- **Language detection** using `langdetect`.
- **Translation support** using `deep-translator` / `googletrans` tools when required.

### Recommendation Logic
The recommendation engine combines multiple factors:
- Patient sentiment from feedback
- Complaint category and semantic similarity
- Doctor rating
- Doctor experience
- Fee and location matching

## Features
- Doctor, specialist, and hospital recommendation
- NLP-based patient review analysis
- Complaint detection from natural language feedback
- Multilingual feedback support
- Fee-based and location-based filtering
- FAQ-based medical query assistance
- Dashboard, login, enrollment, results, and doctor detail pages

## Required Dependencies / Libraries

Main dependencies used in the project:
- Flask
- pandas
- scikit-learn
- openpyxl
- deep-translator
- sentence-transformers
- torch
- numpy
- langdetect
- googletrans
- google-trans-new

Recommended installation files:
- `requirements.txt`
- `requirements_advanced.txt`

## Steps to Run the Project

### 1. Clone or open the folder
Open the project folder in VS Code:
`ml website project`

### 2. Create and activate a Python environment
Example using virtualenv:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
For the standard version:
```bash
pip install -r requirements.txt
```

For the advanced SBERT version:
```bash
pip install -r requirements_advanced.txt
```

### 4. Train the advanced model
Run this if you want SBERT-based multilingual doctor matching:
```bash
python train_advanced.py
```

### 5. Start the Flask app
```bash
python app.py
```

### 6. Open the website
Open your browser and visit:
```text
http://localhost:5000
```

## Sample Output / Screenshots

The repository currently includes output files instead of image screenshots. You can use these as sample result references:
- `outputs/processed_tabular.csv`
- `phase2_outputs/processed_dataset.csv`
- `phase2_outputs/model_comparison.csv`
- `phase2_outputs/summary.json`

Suggested screenshots to capture for submission:
- Home page
- Search/recommendation form
- Doctor recommendation results page
- Doctor profile page
- Multilingual search page

If you add images later, place them in a folder such as `screenshots/` and reference them here.

## Team Member Details

**Team Number:** 14

| Name | Register Number |
|------|-----------------|
| Paramasivam A | 24BCS198 |
| Prawin H | 24BCS209 |
| Rupak Krishna P M | 24BCS231 |
| Santhosh Krishnaa M | 24BCS245 |

## Folder Structure Overview

```text
ml website project/
├── app.py
├── model.py
├── sbert_model.py
├── sbert_model_advanced.py
├── train_sbert.py
├── train_advanced.py
├── inspect_dataset.py
├── languages.py
├── dataset.csv
├── faq.csv
├── requirements.txt
├── requirements_sbert.txt
├── requirements_advanced.txt
├── static/
├── templates/
├── outputs/
├── phase2_outputs/
└── reports/
```

## How the System Helps Patients
- Reduces the effort needed to compare hospitals and doctors.
- Uses patient feedback instead of only star ratings.
- Suggests doctors based on treatment quality, fees, behavior, and waiting time.
- Supports natural language queries and multilingual feedback.

## Limitations
- Recommendation quality depends on dataset quality.
- Multilingual translation accuracy may vary by language.
- Advanced SBERT training may take longer on the first run.

## Future Enhancements
- Add live hospital API integration.
- Add visual dashboards for sentiment trends.
- Expand multilingual support with better translation models.
- Add patient appointment booking and follow-up tracking.

## Quick Demo Commands

```bash
pip install -r requirements_advanced.txt
python train_advanced.py
python app.py
```

## Contact / Submission Note
Before submission, replace the placeholder team member names and add your real screenshots in the report or a `screenshots/` folder.
