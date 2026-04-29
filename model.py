import pandas as pd
import random
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import difflib

DATA_PATH = "Final_20Hospitals_Per_District.xlsx"
FALLBACK_DATA_PATH = "dataset.csv"
DATA_SHEET = "Sheet1"
FAQ_PATH = "faq.csv"

vectorizer = None
sentiment_model = None
label_encoder = None
faq_vectorizer = None
faq_tfidf = None
doctor_examples = []
dataset_condition_guidance = {}
GENERIC_QUERY_WORDS = {
    "what", "should", "know", "about", "disease", "diseases", "doctor",
    "specialist", "condition", "problem", "issue", "symptom", "symptoms",
    "medical", "health", "need", "see", "visit", "for", "i", "my", "a", "an", "the"
}

CONDITION_GUIDANCE = {
    "fever": {
        "keywords": ["fever", "temperature", "viral fever", "body pain with fever"],
        "title": "Fever",
        "possible_reasons": [
            "Viral or bacterial infection",
            "Inflammation in the body",
            "Dehydration or heat exposure",
        ],
        "medicines": [
            "Paracetamol or acetaminophen (Rs. 10-20 per 10 tablets) is commonly used to reduce fever",
            "Ibuprofen (Rs. 15-30 per 10 tablets) may help in some adults if it is safe for them, but avoid it without medical advice if you have kidney problems, ulcers, or are pregnant",
        ],
        "self_care": [
            "Drink plenty of water and rest",
            "Check temperature regularly",
            "Wear light clothing and avoid overheating",
        ],
        "specialist": "General Physician",
        "emergency": [
            "Fever above 103 F or 39.4 C",
            "Breathing trouble, confusion, seizure, or severe weakness",
            "Fever lasting more than 3 days",
        ],
    },
    "cold": {
        "keywords": ["cold", "cough and cold", "runny nose", "common cold"],
        "title": "Common Cold",
        "possible_reasons": [
            "Viral upper respiratory infection",
            "Allergy or weather-triggered irritation",
            "Sinus congestion",
        ],
        "medicines": [
            "Paracetamol can help with fever or body pain",
            "Saline nasal spray or steam inhalation may help nasal blockage",
            "Cough syrup may help depending on the type of cough, but it should be chosen carefully",
        ],
        "self_care": [
            "Take rest and drink warm fluids",
            "Use steam inhalation for congestion relief",
            "Avoid smoking and cold irritants",
        ],
        "specialist": "General Physician or ENT Specialist",
        "emergency": [
            "Shortness of breath",
            "High fever or chest pain",
            "Symptoms worsening after a few days",
        ],
    },
    "headache": {
        "keywords": ["headache", "migraine", "head pain"],
        "title": "Headache",
        "possible_reasons": [
            "Stress or poor sleep",
            "Migraine or eye strain",
            "Dehydration, sinusitis, or infection",
        ],
        "medicines": [
            "Paracetamol (Rs. 10-20 per 10 tablets) is commonly used for mild headache",
            "Ibuprofen (Rs. 15-30 per 10 tablets) may help some people if it is medically safe for them",
        ],
        "self_care": [
            "Rest in a quiet and dark room",
            "Drink water and avoid skipping meals",
            "Reduce screen time if eye strain is suspected",
        ],
        "specialist": "General Physician, Neurologist, or Ophthalmologist depending on symptoms",
        "emergency": [
            "Sudden severe headache",
            "Headache with weakness, fainting, or vision loss",
            "Headache after injury or with persistent vomiting",
        ],
    },
    "stomach pain": {
        "keywords": ["stomach pain", "abdominal pain", "gastric pain", "stomach ache"],
        "title": "Stomach Pain",
        "possible_reasons": [
            "Gas, indigestion, or acidity",
            "Food infection or gastritis",
            "Appendix, gallbladder, or intestinal problems",
        ],
        "medicines": [
            "Antacids like Digene (Rs. 20-40 per bottle) are commonly used for acidity or indigestion",
            "ORS (Rs. 10-20 per packet) may help if there is vomiting or loose motion causing dehydration",
        ],
        "self_care": [
            "Eat light food and avoid oily or spicy meals",
            "Drink water in small amounts",
            "Monitor whether pain is related to food, vomiting, or bowel changes",
        ],
        "specialist": "General Physician or Gastroenterologist",
        "emergency": [
            "Severe or one-sided pain",
            "Blood in stool or vomit",
            "Persistent vomiting, high fever, or rigid abdomen",
        ],
    },
    "tooth pain": {
        "keywords": ["tooth pain", "toothache", "teeth pain", "dental pain"],
        "title": "Tooth Pain",
        "possible_reasons": [
            "Tooth decay or cavity",
            "Gum infection or swelling",
            "Cracked tooth or sensitivity",
        ],
        "medicines": [
            "Paracetamol (Rs. 10-20 per 10 tablets) is commonly used for pain relief",
            "Dentists may prescribe antibiotics like Amoxicillin (Rs. 50-100 per strip) only if there is an infection, so do not start them on your own",
        ],
        "self_care": [
            "Rinse with warm salt water",
            "Avoid very hot, cold, or sugary food",
            "Brush gently and keep the area clean",
        ],
        "specialist": "Dentist",
        "emergency": [
            "Facial swelling",
            "Fever with dental pain",
            "Difficulty swallowing or opening the mouth",
        ],
    },
    "chest pain": {
        "keywords": ["chest pain", "heart pain", "tightness in chest"],
        "title": "Chest Pain",
        "possible_reasons": [
            "Heart-related problem",
            "Acidity or gastric reflux",
            "Muscle strain, anxiety, or lung problem",
        ],
        "medicines": [
            "Do not self-medicate for unexplained chest pain",
            "If you already have doctor-prescribed emergency heart medication, use it exactly as instructed by your doctor",
        ],
        "self_care": [
            "Stop activity and sit down",
            "Seek urgent medical evaluation",
            "Note whether pain spreads to arm, jaw, or back",
        ],
        "specialist": "Cardiologist or Emergency Care",
        "emergency": [
            "Chest pain with sweating, breathlessness, or fainting",
            "Pain spreading to the arm, jaw, or back",
            "Known heart disease with new chest discomfort",
        ],
    },
    "skin rash": {
        "keywords": ["skin rash", "rash", "itching", "allergy", "skin allergy"],
        "title": "Skin Rash or Allergy",
        "possible_reasons": [
            "Allergic reaction",
            "Fungal, bacterial, or viral skin infection",
            "Reaction to soap, food, medicine, or heat",
        ],
        "medicines": [
            "Calamine lotion (Rs. 30-50 per bottle) or mild anti-itch creams may help some simple rashes",
            "Antihistamines like Cetirizine (Rs. 20-40 per 10 tablets) are commonly used for allergy symptoms, but they should be chosen carefully based on age and medical history",
        ],
        "self_care": [
            "Avoid scratching",
            "Use mild soap and keep the area dry",
            "Stop any recently started cosmetic or skin product if it seems to trigger the rash",
        ],
        "specialist": "Dermatologist",
        "emergency": [
            "Swelling of lips or face",
            "Breathing difficulty",
            "Fast-spreading rash, fever, or skin peeling",
        ],
    },
}

CONDITION_GUIDANCE.update({
    "diabetes": {
        "keywords": ["diabetes", "blood sugar", "high sugar", "frequent urination", "excess thirst"],
        "title": "Diabetes",
        "possible_reasons": [
            "High blood glucose due to reduced insulin action",
            "Family history, obesity, or lifestyle factors",
            "Hormonal or metabolic health conditions",
        ],
        "medicines": [
            "Doctors may use metformin or other diabetes medicines depending on the patient",
            "Insulin may be needed in some situations",
        ],
        "self_care": [
            "Monitor blood sugar regularly",
            "Follow a controlled diet and regular exercise plan",
            "Take medicines exactly as prescribed",
        ],
        "specialist": "Diabetologist, Endocrinologist, or General Physician",
        "emergency": [
            "Very high sugar with vomiting or drowsiness",
            "Confusion, dehydration, or difficulty breathing",
            "Repeated low sugar episodes",
        ],
    },
    "hypertension": {
        "keywords": ["bp", "high blood pressure", "hypertension", "pressure problem"],
        "title": "High Blood Pressure",
        "possible_reasons": [
            "Stress, salt-heavy diet, and low physical activity",
            "Kidney, hormone, or heart-related causes",
            "Family history or obesity",
        ],
        "medicines": [
            "Blood pressure tablets must be chosen by a doctor",
            "Do not stop prescribed BP medicines suddenly",
        ],
        "self_care": [
            "Reduce salt intake",
            "Check blood pressure regularly",
            "Exercise, sleep well, and avoid smoking",
        ],
        "specialist": "General Physician or Cardiologist",
        "emergency": [
            "BP very high with severe headache or chest pain",
            "Breathing difficulty or weakness on one side",
            "Blurred vision or confusion",
        ],
    },
    "asthma": {
        "keywords": ["asthma", "wheezing", "breathlessness", "shortness of breath", "chest tightness"],
        "title": "Asthma",
        "possible_reasons": [
            "Airway inflammation and allergy triggers",
            "Dust, smoke, infection, or weather changes",
            "Exercise or strong smells in sensitive people",
        ],
        "medicines": [
            "Inhalers are commonly used for relief and control",
            "Use asthma medicines only as directed by a doctor",
        ],
        "self_care": [
            "Avoid known triggers like dust and smoke",
            "Keep prescribed inhalers available",
            "Seek review if you need the reliever inhaler often",
        ],
        "specialist": "Pulmonologist or General Physician",
        "emergency": [
            "Severe breathlessness or bluish lips",
            "Unable to speak full sentences",
            "Reliever inhaler not helping",
        ],
    },
    "arthritis": {
        "keywords": ["arthritis", "joint pain", "joint swelling", "stiff joints", "swollen joints"],
        "title": "Arthritis or Joint Inflammation",
        "possible_reasons": [
            "Wear-and-tear joint disease",
            "Inflammatory arthritis",
            "Old injury or infection-related joint issues",
        ],
        "medicines": [
            "Pain relief and anti-inflammatory medicines may be used based on the condition",
            "Some patients also need physiotherapy guidance",
        ],
        "self_care": [
            "Avoid overloading painful joints",
            "Use doctor-advised exercises",
            "Track swelling, morning stiffness, and fever",
        ],
        "specialist": "Orthopedic Doctor or Rheumatology Review",
        "emergency": [
            "Very hot swollen joint with fever",
            "Sudden inability to bear weight",
            "Severe deformity or injury",
        ],
    },
    "sinusitis": {
        "keywords": ["sinus", "sinusitis", "nose congestion", "facial pressure", "nasal blockage"],
        "title": "Sinusitis",
        "possible_reasons": [
            "Infection or allergy-related sinus swelling",
            "Cold leading to blocked drainage",
            "Dust or weather-triggered irritation",
        ],
        "medicines": [
            "Saline rinse and steam may help congestion",
            "Other medicines depend on the cause and should be doctor-guided",
        ],
        "self_care": [
            "Use steam inhalation carefully",
            "Drink fluids and rest",
            "Avoid dust and smoke",
        ],
        "specialist": "ENT Specialist",
        "emergency": [
            "Eye swelling or severe facial pain",
            "High fever with worsening symptoms",
            "Symptoms not improving after many days",
        ],
    },
    "dengue": {
        "keywords": ["dengue", "platelet", "mosquito fever", "high fever with body pain"],
        "title": "Dengue",
        "possible_reasons": [
            "Mosquito-borne viral infection",
            "Seasonal outbreaks in some places",
            "Low fluid intake can worsen weakness during illness",
        ],
        "medicines": [
            "Paracetamol may be used for fever only if a doctor says it is suitable",
            "Avoid self-medicating with strong painkillers unless medically advised",
        ],
        "self_care": [
            "Drink plenty of fluids",
            "Monitor weakness, bleeding, and urine output",
            "Follow blood test advice if given by a doctor",
        ],
        "specialist": "General Physician",
        "emergency": [
            "Bleeding, black stool, severe stomach pain, or repeated vomiting",
            "Extreme weakness, fainting, or confusion",
            "Very low urine output",
        ],
    },
    "typhoid": {
        "keywords": ["typhoid", "enteric fever", "prolonged fever", "stomach pain with fever"],
        "title": "Typhoid",
        "possible_reasons": [
            "Bacterial infection from contaminated food or water",
            "Poor sanitation exposure",
            "Delayed treatment can worsen dehydration and weakness",
        ],
        "medicines": [
            "Typhoid needs doctor-prescribed antibiotics",
            "ORS and fluids may help prevent dehydration",
        ],
        "self_care": [
            "Drink safe water and eat light food",
            "Take the full course of prescribed treatment",
            "Rest and monitor fever pattern",
        ],
        "specialist": "General Physician",
        "emergency": [
            "Persistent high fever with confusion",
            "Severe abdominal pain or repeated vomiting",
            "Signs of dehydration",
        ],
    },
    "malaria": {
        "keywords": ["malaria", "fever with chills", "shivering fever", "mosquito bite fever"],
        "title": "Malaria",
        "possible_reasons": [
            "Mosquito-borne parasitic infection",
            "Travel or stay in affected areas",
            "Delayed treatment can increase severity",
        ],
        "medicines": [
            "Malaria treatment depends on test results and medical evaluation",
            "Do not start medicines without diagnosis confirmation",
        ],
        "self_care": [
            "Get tested if fever with chills keeps returning",
            "Drink fluids and rest",
            "Monitor fever timing and weakness",
        ],
        "specialist": "General Physician",
        "emergency": [
            "Confusion, breathing trouble, or severe weakness",
            "Persistent vomiting or seizures",
            "Very high fever not settling",
        ],
    },
    "pneumonia": {
        "keywords": ["pneumonia", "lung infection", "cough with fever", "chest infection"],
        "title": "Pneumonia",
        "possible_reasons": [
            "Bacterial, viral, or fungal lung infection",
            "Weak immunity or older age can increase risk",
            "Untreated infection may affect breathing",
        ],
        "medicines": [
            "Treatment depends on the cause and severity",
            "Doctor-guided antibiotics may be needed in some cases",
        ],
        "self_care": [
            "Rest well and drink fluids",
            "Monitor cough, fever, and breathing",
            "Do not ignore chest pain or breathlessness",
        ],
        "specialist": "Pulmonologist or General Physician",
        "emergency": [
            "Shortness of breath or bluish lips",
            "Chest pain with low oxygen symptoms",
            "Confusion or very high fever",
        ],
    },
    "bronchitis": {
        "keywords": ["bronchitis", "persistent cough", "chest congestion", "mucus cough"],
        "title": "Bronchitis",
        "possible_reasons": [
            "Airway infection or inflammation",
            "Smoking, pollution, or viral illness",
            "Sometimes follows a cold or flu",
        ],
        "medicines": [
            "Treatment depends on whether the cause is viral, bacterial, or allergic",
            "Cough medicines should be chosen carefully",
        ],
        "self_care": [
            "Rest and drink warm fluids",
            "Avoid smoking and dust exposure",
            "Monitor for wheezing or breathlessness",
        ],
        "specialist": "General Physician or Pulmonologist",
        "emergency": [
            "Breathing trouble",
            "High fever with chest pain",
            "Cough worsening significantly",
        ],
    },
    "tuberculosis": {
        "keywords": ["tb", "tuberculosis", "cough for weeks", "weight loss with cough"],
        "title": "Tuberculosis",
        "possible_reasons": [
            "Bacterial infection affecting the lungs or other organs",
            "Close contact with an infected person",
            "Low immunity can increase risk",
        ],
        "medicines": [
            "TB needs confirmed diagnosis and a full doctor-prescribed treatment course",
            "Do not stop treatment early if it is started",
        ],
        "self_care": [
            "Get tested for chronic cough, fever, or weight loss",
            "Maintain nutrition and follow-up regularly",
            "Take medicines exactly as prescribed",
        ],
        "specialist": "Chest Physician or General Physician",
        "emergency": [
            "Breathing trouble or coughing blood",
            "Severe weakness or persistent fever",
            "Unable to eat or drink properly",
        ],
    },
    "kidney stone": {
        "keywords": ["kidney stone", "renal stone", "side pain", "stone pain", "urine stone"],
        "title": "Kidney Stone",
        "possible_reasons": [
            "Mineral crystal formation in the urinary tract",
            "Low water intake",
            "Dietary and metabolic factors",
        ],
        "medicines": [
            "Pain relief and other treatment depend on the stone size and location",
            "Some cases need procedures by a specialist",
        ],
        "self_care": [
            "Drink enough water unless a doctor restricts fluids",
            "Track pain, fever, and urine symptoms",
            "Seek review if pain is severe or recurrent",
        ],
        "specialist": "Urologist or General Physician",
        "emergency": [
            "Severe side pain with vomiting",
            "Fever with urine blockage symptoms",
            "Blood in urine with severe weakness",
        ],
    },
    "uti": {
        "keywords": ["uti", "urine infection", "burning urination", "frequent urination", "urinary infection"],
        "title": "Urinary Tract Infection",
        "possible_reasons": [
            "Bacterial infection in the urinary tract",
            "Low fluid intake or delayed urination",
            "Sometimes associated with stones or diabetes",
        ],
        "medicines": [
            "UTI often needs doctor-advised antibiotics",
            "Fluids may help but do not replace treatment",
        ],
        "self_care": [
            "Drink water regularly",
            "Do not hold urine for long periods",
            "Seek testing if symptoms continue",
        ],
        "specialist": "General Physician or Urologist",
        "emergency": [
            "Fever with back pain",
            "Vomiting or severe weakness",
            "Blood in urine or severe pain",
        ],
    },
    "appendicitis": {
        "keywords": ["appendicitis", "right lower stomach pain", "appendix pain", "lower abdomen pain"],
        "title": "Appendicitis",
        "possible_reasons": [
            "Inflammation or infection of the appendix",
            "Pain often worsens over time and may shift location",
            "Delayed treatment can be risky",
        ],
        "medicines": [
            "Do not self-treat suspected appendicitis at home",
            "Treatment usually needs urgent doctor evaluation",
        ],
        "self_care": [
            "Avoid delaying hospital review",
            "Track fever, vomiting, and pain location",
            "Do not eat heavily if surgery is being considered",
        ],
        "specialist": "General Surgeon or Emergency Care",
        "emergency": [
            "Severe stomach pain with fever or vomiting",
            "Pain getting worse quickly",
            "Rigid abdomen or difficulty walking due to pain",
        ],
    },
    "gastritis": {
        "keywords": ["gastritis", "acidity", "burning stomach", "indigestion", "stomach burning"],
        "title": "Gastritis or Acidity",
        "possible_reasons": [
            "Acid irritation in the stomach lining",
            "Spicy food, irregular meals, alcohol, or certain medicines",
            "Infection or stress can also contribute",
        ],
        "medicines": [
            "Antacids or acid-reducing medicines may be used if a doctor feels they are suitable",
            "Avoid overusing pain tablets that irritate the stomach",
        ],
        "self_care": [
            "Eat light meals and avoid spicy or oily food",
            "Do not skip meals",
            "Reduce alcohol and smoking",
        ],
        "specialist": "General Physician or Gastroenterologist",
        "emergency": [
            "Black stool or vomiting blood",
            "Severe persistent stomach pain",
            "Repeated vomiting or dehydration",
        ],
    },
    "ulcer": {
        "keywords": ["ulcer", "stomach ulcer", "gastric ulcer", "peptic ulcer"],
        "title": "Peptic Ulcer",
        "possible_reasons": [
            "Acid injury in the stomach or upper intestine",
            "Certain medicines or bacterial infection",
            "Smoking and alcohol may worsen symptoms",
        ],
        "medicines": [
            "Acid-reducing medicines are often used under medical advice",
            "Treatment may also depend on infection testing",
        ],
        "self_care": [
            "Avoid spicy food, smoking, and alcohol",
            "Take medicines regularly if prescribed",
            "Seek review for persistent pain or vomiting",
        ],
        "specialist": "Gastroenterologist or General Physician",
        "emergency": [
            "Vomiting blood or black stool",
            "Sudden severe abdominal pain",
            "Severe weakness or fainting",
        ],
    },
    "eczema": {
        "keywords": ["eczema", "dry itchy skin", "skin patch itching", "dermatitis"],
        "title": "Eczema or Dermatitis",
        "possible_reasons": [
            "Skin barrier irritation and inflammation",
            "Allergy, dryness, soap, or weather triggers",
            "Family history can play a role",
        ],
        "medicines": [
            "Moisturizers and doctor-advised skin creams are commonly used",
            "Avoid random steroid creams without guidance",
        ],
        "self_care": [
            "Use mild soap and moisturize regularly",
            "Avoid scratching and trigger products",
            "Keep nails short if itching is severe",
        ],
        "specialist": "Dermatologist",
        "emergency": [
            "Spreading infection with pus or fever",
            "Severe swelling or skin cracking",
            "Breathing trouble after a trigger",
        ],
    },
    "psoriasis": {
        "keywords": ["psoriasis", "scaly skin patches", "thick skin patch"],
        "title": "Psoriasis",
        "possible_reasons": [
            "Chronic inflammatory skin condition",
            "Immune-related process",
            "Stress and skin injury may worsen flares",
        ],
        "medicines": [
            "Topical creams and other treatments depend on severity",
            "Doctor review is important before starting long-term treatment",
        ],
        "self_care": [
            "Moisturize the skin regularly",
            "Avoid harsh skin products",
            "Track flare triggers like stress or infection",
        ],
        "specialist": "Dermatologist",
        "emergency": [
            "Widespread painful skin inflammation",
            "Joint pain with severe skin flare",
            "Fever with skin worsening",
        ],
    },
    "conjunctivitis": {
        "keywords": ["conjunctivitis", "eye infection", "red eye", "itchy eye", "watering eye"],
        "title": "Eye Infection or Conjunctivitis",
        "possible_reasons": [
            "Viral or bacterial eye infection",
            "Allergy or irritant exposure",
            "Poor eye hygiene or contact lens issues",
        ],
        "medicines": [
            "Artificial tears may soothe irritation",
            "Other eye drops must be chosen carefully by a doctor",
        ],
        "self_care": [
            "Avoid rubbing the eyes",
            "Wash hands often and use a clean towel",
            "Avoid sharing cosmetics or eye drops",
        ],
        "specialist": "Ophthalmologist",
        "emergency": [
            "Vision loss or severe eye pain",
            "Marked swelling or light sensitivity",
            "Symptoms getting worse in contact lens users",
        ],
    },
    "cataract": {
        "keywords": ["cataract", "blurred vision in old age", "cloudy vision"],
        "title": "Cataract",
        "possible_reasons": [
            "Lens clouding in the eye, often age-related",
            "Diabetes, injury, or some medicines can contribute",
            "Vision gradually becomes cloudy or dim",
        ],
        "medicines": [
            "Medicines do not usually remove cataract",
            "Eye specialist evaluation is needed to plan treatment",
        ],
        "self_care": [
            "Use proper glasses if advised",
            "Avoid driving if vision becomes unsafe",
            "Get regular eye checkups",
        ],
        "specialist": "Ophthalmologist",
        "emergency": [
            "Sudden major vision drop",
            "Eye pain with blurred vision",
            "Injury-related eye symptoms",
        ],
    },
    "glaucoma": {
        "keywords": ["glaucoma", "eye pressure", "vision field loss", "eye nerve problem"],
        "title": "Glaucoma",
        "possible_reasons": [
            "Raised eye pressure damaging the optic nerve",
            "Family history can increase risk",
            "Often needs early diagnosis to protect vision",
        ],
        "medicines": [
            "Eye drops and other treatments must be prescribed by an eye specialist",
            "Do not skip regular follow-up if diagnosed",
        ],
        "self_care": [
            "Attend eye pressure checkups regularly",
            "Take drops exactly as prescribed",
            "Report any sudden eye pain or vision changes",
        ],
        "specialist": "Ophthalmologist",
        "emergency": [
            "Sudden severe eye pain",
            "Halos around lights with vomiting",
            "Rapid vision loss",
        ],
    },
    "anemia": {
        "keywords": ["anemia", "low hemoglobin", "pale skin", "tiredness", "weakness"],
        "title": "Anemia",
        "possible_reasons": [
            "Low hemoglobin due to iron deficiency or other causes",
            "Blood loss, poor diet, or chronic disease",
            "Sometimes linked with vitamin deficiency",
        ],
        "medicines": [
            "Treatment depends on the cause after testing",
            "Iron or vitamin supplements should be used only if appropriate",
        ],
        "self_care": [
            "Get blood tests if weakness is persistent",
            "Eat iron-rich foods if medically suitable",
            "Follow treatment and repeat testing if advised",
        ],
        "specialist": "General Physician",
        "emergency": [
            "Very severe weakness or shortness of breath",
            "Heavy bleeding",
            "Fainting or chest discomfort",
        ],
    },
    "thyroid disorder": {
        "keywords": ["thyroid", "thyroid disorder", "neck gland problem", "hormone imbalance"],
        "title": "Thyroid Disorder",
        "possible_reasons": [
            "Low or high thyroid hormone levels",
            "Autoimmune or gland-related conditions",
            "Symptoms vary from weight change to tiredness or palpitations",
        ],
        "medicines": [
            "Thyroid treatment depends on blood test results",
            "Dose adjustment should be done only by a doctor",
        ],
        "self_care": [
            "Get thyroid function tests if advised",
            "Take medicines consistently if prescribed",
            "Track weight, energy level, and heart symptoms",
        ],
        "specialist": "Endocrinologist or General Physician",
        "emergency": [
            "Severe palpitations or confusion",
            "Extreme weakness or breathlessness",
            "Rapid symptom worsening",
        ],
    },
    "depression": {
        "keywords": ["depression", "low mood", "sadness", "loss of interest", "mental health"],
        "title": "Depression",
        "possible_reasons": [
            "Mental health condition affecting mood and daily function",
            "Stress, trauma, biology, or long-term illness can contribute",
            "Sleep and appetite changes may also occur",
        ],
        "medicines": [
            "Treatment may involve counseling, therapy, and sometimes medicines",
            "Mental health medicines must be doctor-guided",
        ],
        "self_care": [
            "Seek support early instead of waiting",
            "Maintain sleep, food, and routine as much as possible",
            "Talk to a trusted person or professional",
        ],
        "specialist": "Psychiatrist or Clinical Psychologist",
        "emergency": [
            "Self-harm thoughts or suicidal thoughts",
            "Severe inability to function",
            "Confusion or extreme agitation",
        ],
    },
    "anxiety": {
        "keywords": ["anxiety", "panic", "panic attack", "excess worry", "nervousness"],
        "title": "Anxiety or Panic Symptoms",
        "possible_reasons": [
            "Stress response or anxiety disorder",
            "Sleep problems, caffeine, or emotional triggers",
            "Some physical illnesses can mimic anxiety symptoms",
        ],
        "medicines": [
            "Treatment may include counseling and sometimes medicines",
            "Do not self-medicate with sedatives",
        ],
        "self_care": [
            "Try slow breathing and reduce stimulant intake",
            "Track triggers and sleep pattern",
            "Seek mental health support if symptoms recur",
        ],
        "specialist": "Psychiatrist, Psychologist, or General Physician",
        "emergency": [
            "Chest pain or breathing trouble needing immediate evaluation",
            "Self-harm thoughts",
            "Severe confusion or uncontrolled panic",
        ],
    },
    "covid-19": {
        "keywords": ["covid", "coronavirus", "covid-19", "pandemic fever", "loss of taste"],
        "title": "COVID-19",
        "possible_reasons": [
            "Viral infection caused by SARS-CoV-2",
            "Spread through respiratory droplets",
            "Symptoms can range from mild to severe",
        ],
        "medicines": [
            "Treatment depends on severity; some cases need hospitalization",
            "Vaccines and preventive measures are key",
        ],
        "self_care": [
            "Isolate if symptoms appear",
            "Wear masks and maintain hygiene",
            "Monitor fever, cough, and breathing",
        ],
        "specialist": "General Physician or Infectious Disease Specialist",
        "emergency": [
            "Difficulty breathing or chest pain",
            "High fever with confusion",
            "Blue lips or severe weakness",
        ],
    },
    "cancer": {
        "keywords": ["cancer", "tumor", "malignancy", "oncology", "chemo"],
        "title": "Cancer",
        "possible_reasons": [
            "Uncontrolled cell growth leading to tumors",
            "Genetic, environmental, or lifestyle factors",
            "Early detection improves outcomes",
        ],
        "medicines": [
            "Treatment includes surgery, chemotherapy, radiation, or targeted therapy",
            "Always under oncologist guidance",
        ],
        "self_care": [
            "Follow screening recommendations",
            "Maintain healthy lifestyle",
            "Seek support for emotional well-being",
        ],
        "specialist": "Oncologist",
        "emergency": [
            "Severe pain or bleeding",
            "Difficulty breathing or swallowing",
            "Sudden worsening of symptoms",
        ],
    },
    "stroke": {
        "keywords": ["stroke", "brain attack", "paralysis", "sudden weakness", "speech loss"],
        "title": "Stroke",
        "possible_reasons": [
            "Blockage or rupture of blood vessels in the brain",
            "High blood pressure, diabetes, or smoking increases risk",
            "Requires immediate medical attention",
        ],
        "medicines": [
            "Emergency treatment like clot-busting drugs if applicable",
            "Long-term prevention with blood thinners or BP control",
        ],
        "self_care": [
            "Control risk factors like BP and diabetes",
            "Exercise and healthy diet",
            "Recognize FAST signs: Face drooping, Arm weakness, Speech slurred, Time to call emergency",
        ],
        "specialist": "Neurologist or Emergency Care",
        "emergency": [
            "Sudden numbness or weakness",
            "Confusion or trouble speaking",
            "Vision problems or difficulty walking",
        ],
    },
    "kidney disease": {
        "keywords": ["kidney disease", "renal failure", "chronic kidney", "dialysis"],
        "title": "Kidney Disease",
        "possible_reasons": [
            "Damage to kidney function over time",
            "Diabetes, hypertension, or infections",
            "May lead to fluid retention or waste buildup",
        ],
        "medicines": [
            "BP and diabetes control are key",
            "Some cases need dialysis or transplant",
        ],
        "self_care": [
            "Monitor urine output and swelling",
            "Low salt diet if advised",
            "Regular checkups for kidney function",
        ],
        "specialist": "Nephrologist",
        "emergency": [
            "Severe swelling or shortness of breath",
            "Blood in urine or severe pain",
            "Confusion or seizures",
        ],
    },
    "liver disease": {
        "keywords": ["liver disease", "hepatitis", "jaundice", "liver cirrhosis", "fatty liver"],
        "title": "Liver Disease",
        "possible_reasons": [
            "Infection, alcohol, or fatty accumulation",
            "Hepatitis viruses or autoimmune issues",
            "Can lead to jaundice or fluid buildup",
        ],
        "medicines": [
            "Treatment depends on the cause",
            "Vaccines for hepatitis if applicable",
        ],
        "self_care": [
            "Avoid alcohol and maintain healthy weight",
            "Get vaccinated for hepatitis",
            "Monitor for jaundice or abdominal swelling",
        ],
        "specialist": "Hepatologist or Gastroenterologist",
        "emergency": [
            "Severe abdominal pain or vomiting blood",
            "Confusion or jaundice with fever",
            "Swelling with breathing difficulty",
        ],
    },
    "migraine": {
        "keywords": ["migraine", "severe headache", "throbbing head pain", "headache with nausea"],
        "title": "Migraine",
        "possible_reasons": [
            "Neurological condition causing intense headaches",
            "Triggers like stress, food, or hormones",
            "Family history common",
        ],
        "medicines": [
            "Pain relievers or specific migraine medicines",
            "Preventive treatment if frequent",
        ],
        "self_care": [
            "Identify and avoid triggers",
            "Rest in dark room during attack",
            "Maintain regular sleep and meals",
        ],
        "specialist": "Neurologist",
        "emergency": [
            "Headache with neurological symptoms like weakness",
            "Sudden severe headache",
            "Headache after head injury",
        ],
    },
    "epilepsy": {
        "keywords": ["epilepsy", "seizure", "fits", "convulsions", "epileptic"],
        "title": "Epilepsy",
        "possible_reasons": [
            "Abnormal brain electrical activity",
            "Head injury, infection, or genetic factors",
            "Seizures can vary in type",
        ],
        "medicines": [
            "Anti-seizure medicines to control episodes",
            "Dose adjustment by neurologist",
        ],
        "self_care": [
            "Take medicines regularly",
            "Avoid triggers like lack of sleep",
            "Wear medical ID bracelet",
        ],
        "specialist": "Neurologist",
        "emergency": [
            "Prolonged seizure or multiple seizures",
            "Injury during seizure",
            "First seizure or status epilepticus",
        ],
    },
})

PART_TO_SPECIALIZATIONS = {
    "eyes": [
        "Ophthalmologist",
        "Optometrist",
        "Ophthalmology",
        "Opthalmology",
        "Eye",
        "Eye Specialist",
        "Eye Care",
    ],
    "eye": [
        "Ophthalmologist",
        "Optometrist",
        "Ophthalmology",
        "Opthalmology",
        "Eye",
        "Eye Specialist",
        "Eye Care",
    ],
    "nose": ["ENT", "ENT Specialist"],
    "ent": ["ENT", "ENT Specialist"],
    "teeth": ["Dentist", "Dental"],
    "tooth": ["Dentist", "Dental"],
    "skin": ["Dermatologist"],
    "skin-hair": ["Dermatologist"],
    "bones": ["Orthopedic", "Orthopaedic"],
    "bone": ["Orthopedic", "Orthopaedic"],
    "heart": ["Cardiologist", "Cardiology"],
    "liver": ["Hepatologist", "Gastroenterologist"],
    "lungs": ["Pulmonologist", "Chest Physician"],
    "brain": ["Neurologist", "Neurosurgeon"],
    "neuro": ["Neurologist", "Neurology"],
    "kidney": ["Urologist", "Nephrologist"],
    "general": ["General Physician", "Physician"],
}

# List of all disease names for auto-correction
DISEASE_NAMES = [
    "fever", "cold", "headache", "stomach pain", "tooth pain", "chest pain", "skin rash",
    "diabetes", "hypertension", "asthma", "arthritis", "sinusitis", "dengue", "typhoid",
    "malaria", "pneumonia", "bronchitis", "tuberculosis", "kidney stone", "uti",
    "appendicitis", "gastritis", "ulcer", "eczema", "psoriasis", "conjunctivitis",
    "cataract", "glaucoma", "anemia", "thyroid disorder", "depression", "anxiety",
    "covid-19", "cancer", "stroke", "kidney disease", "liver disease", "migraine", "epilepsy",
    "coronavirus", "brain attack", "renal failure", "chronic kidney", "hepatitis", "jaundice",
    "liver cirrhosis", "fatty liver", "severe headache", "throbbing head pain", "headache with nausea",
    "seizure", "fits", "convulsions", "epileptic"
]

DISEASE_TO_PART = {
    "conjunctivitis": "eye",
    "cataract": "eye",
    "glaucoma": "eye",
    "dry eye": "eye",
    "macular": "eye",
    "ear pain": "ent",
    "ear": "ent",
    "sinusitis": "nose",
    "ear infection": "ent",
    "tonsillitis": "ent",
    "tooth": "teeth",
    "cavities": "teeth",
    "gum": "teeth",
    "fracture": "bones",
    "arthritis": "bones",
    "osteoporosis": "bones",
    "eczema": "skin",
    "psoriasis": "skin",
    "acne": "skin",
    "asthma": "lungs",
    "pneumonia": "lungs",
    "tuberculosis": "lungs",
    "copd": "lungs",
    "heart": "heart",
    "cardiac": "heart",
    "kidney": "kidney",
    "uti": "kidney",
    "liver": "liver",
    "hepatitis": "liver",
    "stroke": "brain",
    "migraine": "brain",
    "epilepsy": "brain",
}

BODY_PART_GUIDANCE = {
    "eye": {
        "keywords": ["eye", "eyes", "vision", "sight", "blind", "glasses"],
        "title": "Eye Problems",
        "possible_reasons": [
            "Eye strain, infection, allergy, or vision-related problems",
            "Dry eyes, conjunctivitis, cataract, or glaucoma depending on symptoms",
        ],
        "medicines": [
            "Use only doctor-advised eye drops like artificial tears (Rs. 50-100 per bottle)",
            "Avoid self-medicating with steroid eye drops",
        ],
        "self_care": [
            "Avoid rubbing the eyes",
            "Reduce screen strain and keep eyes clean",
        ],
        "specialist": "Ophthalmologist",
        "emergency": [
            "Sudden vision loss",
            "Severe eye pain, swelling, or injury",
        ],
        "faq_answer": "Eye symptoms can be related to infection, allergy, strain, cataract, or pressure problems. An ophthalmologist is the right specialist for proper evaluation.",
    },
    "nose": {
        "keywords": ["nose", "nasal", "sinus", "congestion", "runny nose", "blocked nose"],
        "title": "Nose Problems",
        "possible_reasons": [
            "Cold, sinusitis, allergy, or infection",
            "Nose block, bleeding, or facial pressure may need ENT review",
        ],
        "medicines": [
            "Saline rinse (Rs. 20-40 per bottle) or steam may help some simple congestion",
            "Other medicines depend on the exact cause",
        ],
        "self_care": [
            "Avoid dust and smoke exposure",
            "Stay hydrated and monitor fever or discharge",
        ],
        "specialist": "ENT Specialist",
        "emergency": [
            "Heavy bleeding",
            "Breathing trouble or severe facial pain",
        ],
        "faq_answer": "Nose complaints are commonly due to allergy, cold, sinusitis, or bleeding problems. An ENT specialist can evaluate blocked nose, discharge, or repeated nosebleeds.",
    },
    "skin": {
        "keywords": ["skin", "rash", "itching", "eczema", "dermatitis", "allergy"],
        "title": "Skin Problems",
        "possible_reasons": [
            "Allergy, rash, eczema, fungal infection, or irritation",
            "Itching, redness, scaling, or swelling can have multiple causes",
        ],
        "medicines": [
            "Use only suitable creams after medical advice if symptoms are significant",
            "Avoid random steroid creams",
        ],
        "self_care": [
            "Keep skin clean and dry",
            "Avoid scratching and avoid trigger products",
        ],
        "specialist": "Dermatologist",
        "emergency": [
            "Breathing trouble with rash",
            "Fast-spreading rash, swelling, or skin peeling",
        ],
        "faq_answer": "Skin symptoms can happen due to allergy, infection, eczema, or psoriasis. A dermatologist is the right specialist if the rash is persistent, spreading, or very itchy.",
    },
    "heart": {
        "keywords": ["heart", "chest pain", "palpitations", "cardiac", "blood pressure"],
        "title": "Heart Problems",
        "possible_reasons": [
            "Heart-related chest pain, rhythm issues, or blood pressure problems",
            "Symptoms like chest pain, breathlessness, or dizziness need attention",
        ],
        "medicines": [
            "Do not self-medicate for chest pain or suspected heart symptoms",
            "Use prescribed heart medicines only as instructed by your doctor",
        ],
        "self_care": [
            "Stop activity and rest if symptoms start",
            "Seek medical review quickly for chest discomfort or palpitations",
        ],
        "specialist": "Cardiologist",
        "emergency": [
            "Chest pain with sweating or breathlessness",
            "Fainting or pain spreading to the arm, jaw, or back",
        ],
        "faq_answer": "Heart-related complaints should be checked by a cardiologist, especially if there is chest pain, palpitations, swelling in legs, or shortness of breath.",
    },
    "teeth": {
        "keywords": ["teeth", "tooth", "dental", "mouth", "gum", "pain"],
        "title": "Teeth or Dental Problems",
        "possible_reasons": [
            "Tooth decay, gum infection, sensitivity, or dental injury",
            "Pain, swelling, or bleeding gums may need dental care",
        ],
        "medicines": [
            "Simple pain relief like Paracetamol (Rs. 10-20 per 10 tablets) may help temporarily if it is safe for you",
            "Antibiotics should be used only if a dentist prescribes them",
        ],
        "self_care": [
            "Brush gently and rinse with warm salt water",
            "Avoid very hot, cold, or sugary foods",
        ],
        "specialist": "Dentist",
        "emergency": [
            "Facial swelling",
            "Difficulty swallowing or opening the mouth",
        ],
        "faq_answer": "Dental complaints often come from cavities, gum infection, or sensitivity. A dentist is the correct specialist for pain, swelling, or damaged teeth.",
    },
    "bone": {
        "keywords": ["bone", "joint", "arthritis", "fracture", "pain", "swelling"],
        "title": "Bone or Joint Problems",
        "possible_reasons": [
            "Fracture, arthritis, injury, or swelling-related conditions",
            "Bone pain or movement difficulty may need orthopedic review",
        ],
        "medicines": [
            "Pain relief like Ibuprofen (Rs. 15-30 per 10 tablets) depends on the exact injury or condition",
            "Some cases may need scans, support, or procedures",
        ],
        "self_care": [
            "Avoid stressing the painful area",
            "Seek review if swelling or movement restriction is present",
        ],
        "specialist": "Orthopedic Doctor",
        "emergency": [
            "Visible deformity",
            "Unable to move or bear weight after injury",
        ],
        "faq_answer": "Bone and joint symptoms are commonly checked by an orthopedic doctor, especially if there is pain, swelling, deformity, or difficulty moving.",
    },
    "brain": {
        "keywords": ["brain", "headache", "migraine", "neurological", "dizziness", "stroke"],
        "title": "Brain or Neurological Problems",
        "possible_reasons": [
            "Headache, migraine, stroke, or neurological disorders",
            "Symptoms like dizziness, confusion, or weakness may need neurology review",
        ],
        "medicines": [
            "Medicines depend on the exact diagnosis",
            "Do not self-medicate for neurological symptoms",
        ],
        "self_care": [
            "Rest and avoid stress triggers",
            "Monitor symptoms and seek review if worsening",
        ],
        "specialist": "Neurologist",
        "emergency": [
            "Sudden weakness or numbness",
            "Severe headache or confusion",
        ],
        "faq_answer": "Brain-related symptoms can be due to migraine, stroke, or other conditions. A neurologist is the right specialist for headaches, dizziness, or neurological issues.",
    },
    "stomach": {
        "keywords": ["stomach", "abdominal", "vomit", "vomiting", "nausea", "gastritis", "acidity"],
        "title": "Stomach or Digestive Problems",
        "possible_reasons": [
            "Acidity, gastritis, infection, or digestive disorders",
            "Pain, bloating, or nausea may need gastroenterology review",
        ],
        "medicines": [
            "Antacids like Pantoprazole (Rs. 50-100 per strip) or other medicines depend on the cause",
            "Avoid self-medicating for stomach issues",
        ],
        "self_care": [
            "Eat light meals and avoid spicy food",
            "Stay hydrated and monitor symptoms",
        ],
        "specialist": "Gastroenterologist",
        "emergency": [
            "Severe pain or vomiting blood",
            "Inability to eat or drink",
        ],
        "faq_answer": "Stomach problems can be due to acidity, infection, or ulcers. A gastroenterologist can evaluate pain, bloating, or digestive symptoms.",
    },
    "kidney": {
        "keywords": ["kidney", "urine", "urination", "stone", "pain", "infection"],
        "title": "Kidney Problems",
        "possible_reasons": [
            "Infection, stones, or chronic kidney disease",
            "Symptoms like pain, frequent urination, or swelling may need urology review",
        ],
        "medicines": [
            "Medicines depend on the exact condition",
            "Fluids and diet changes may be advised",
        ],
        "self_care": [
            "Stay hydrated and monitor urine output",
            "Seek review for persistent pain or changes",
        ],
        "specialist": "Urologist or Nephrologist",
        "emergency": [
            "Severe pain or blood in urine",
            "Reduced urine output or swelling",
        ],
        "faq_answer": "Kidney issues can include stones, infection, or failure. A urologist or nephrologist is the specialist for pain, urination problems, or kidney function concerns.",
    },
    "liver": {
        "keywords": ["liver", "jaundice", "fatigue", "hepatitis", "alcohol"],
        "title": "Liver Problems",
        "possible_reasons": [
            "Infection, fatty liver, or liver disease",
            "Symptoms like jaundice, pain, or fatigue may need hepatology review",
        ],
        "medicines": [
            "Treatment depends on the cause",
            "Avoid alcohol and certain medicines",
        ],
        "self_care": [
            "Eat a healthy diet and exercise",
            "Monitor for jaundice or swelling",
        ],
        "specialist": "Hepatologist or Gastroenterologist",
        "emergency": [
            "Severe pain or jaundice",
            "Confusion or bleeding",
        ],
        "faq_answer": "Liver problems can be due to infection, alcohol, or disease. A hepatologist can evaluate jaundice, pain, or liver function abnormalities.",
    },
}


def correct_disease_name(query_text):
    """
    Correct misspelled disease names using difflib.
    """
    query_lower = query_text.lower()
    words = re.findall(r'\b\w+\b', query_lower)
    corrected_words = []
    for word in words:
        if word in DISEASE_NAMES:
            corrected_words.append(word)
        else:
            matches = difflib.get_close_matches(word, DISEASE_NAMES, n=1, cutoff=0.8)
            if matches:
                corrected_words.append(matches[0])
            else:
                corrected_words.append(word)
    return ' '.join(corrected_words)


def _safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _first_symptom(symptom_text):
    parts = [item.strip() for item in _safe_text(symptom_text).split(",") if item.strip()]
    return parts[0].title() if parts else ""


def _top_terms(series, limit=5):
    counter = Counter()
    for value in series.dropna().astype(str):
        for item in value.split(","):
            item = item.strip()
            if item:
                counter[item] += 1
    return [term for term, _ in counter.most_common(limit)]


def _normalize_dataset(raw_df):
    data_df = raw_df.copy().rename(columns={
        "doctor": "doctor_name",
        "disease": "department",
        "symptoms": "symptom_text",
        "review_english": "feedback_text",
        "sentiment": "sentiment_label",
        "rating": "average_rating",
        "Location (Tamil Nadu)": "location",
    })

    if "doctor_name" not in data_df.columns:
        data_df["doctor_name"] = ""
    if "department" not in data_df.columns:
        data_df["department"] = ""
    if "symptom_text" not in data_df.columns:
        data_df["symptom_text"] = data_df.get("complaint_category", "")
    if "feedback_text" not in data_df.columns:
        data_df["feedback_text"] = data_df.get("review_original", data_df.get("feedback_text", ""))
    if "sentiment_label" not in data_df.columns:
        data_df["sentiment_label"] = data_df.get("sentiment_label", "Neutral")
    if "average_rating" not in data_df.columns:
        data_df["average_rating"] = data_df.get("average_rating", 0)
    if "location" not in data_df.columns:
        data_df["location"] = data_df.get("location", "")
    if "body_part" not in data_df.columns:
        data_df["body_part"] = data_df.get("department", "")
    if "complaint_category" not in data_df.columns:
        if "category" in data_df.columns:
            data_df["complaint_category"] = data_df["category"]
        else:
            data_df["complaint_category"] = data_df["symptom_text"].apply(_first_symptom)
    if "qualification" not in data_df.columns:
        data_df["qualification"] = data_df["specialization"].fillna("").astype(str) + " Specialist"
    rng = random.Random(42)
    if "experience_years" not in data_df.columns:
        data_df["experience_years"] = [rng.randint(1, 70) for _ in range(len(data_df))]
    if "hospital" not in data_df.columns:
        data_df["hospital"] = ""
    if "recommendation_label" not in data_df.columns:
        data_df["recommendation_label"] = 0

    for column in [
        "doctor_name",
        "department",
        "symptom_text",
        "feedback_text",
        "sentiment_label",
        "location",
        "specialization",
        "complaint_category",
        "qualification",
        "hospital",
        "body_part",
    ]:
        data_df[column] = data_df[column].fillna("").astype(str)

    if "experience_years" in data_df.columns:
        exp_series = pd.to_numeric(data_df["experience_years"], errors="coerce")
        missing_mask = exp_series.isna()
        if missing_mask.any():
            exp_series.loc[missing_mask] = [rng.randint(1, 70) for _ in range(missing_mask.sum())]
        data_df["experience_years"] = exp_series.astype(int)

    data_df["consultation_fee"] = pd.to_numeric(data_df["consultation_fee"], errors="coerce").fillna(0)
    data_df["average_rating"] = pd.to_numeric(data_df["average_rating"], errors="coerce").fillna(0)
    data_df["recommendation_label"] = pd.to_numeric(data_df["recommendation_label"], errors="coerce").fillna(0)
    data_df = data_df.reset_index(drop=True)
    if "doctor_id" not in data_df.columns:
        data_df["doctor_id"] = data_df.index.astype(int)
    return data_df


def _ensure_unique_doctor_names(data_df):
    if "doctor_name" not in data_df.columns:
        return data_df

    names = data_df["doctor_name"].fillna("").astype(str)
    name_counts = names.groupby(names).cumcount()
    has_name = names.str.len() > 0
    needs_suffix = has_name & (name_counts > 0)
    data_df = data_df.copy()
    data_df.loc[needs_suffix, "doctor_name"] = (
        names[needs_suffix] + " (" + (name_counts[needs_suffix] + 1).astype(str) + ")"
    )
    return data_df


def _prepare_dynamic_resources(data_df):
    global doctor_examples, dataset_condition_guidance
    doctor_examples = (
        data_df.sort_values(by=["average_rating", "recommendation_label"], ascending=False)
        .drop_duplicates(subset=["doctor_name"])
        .head(5)["doctor_name"]
        .tolist()
    )
    dataset_condition_guidance = {}
    for disease, group in data_df.groupby("department"):
        disease_name = _safe_text(disease)
        if not disease_name:
            continue
        symptoms = _top_terms(group["symptom_text"], limit=5)
        specializations = [item for item in group["specialization"].dropna().astype(str).unique().tolist()[:2] if item]
        locations = [item for item in group["location"].dropna().astype(str).unique().tolist()[:3] if item]
        dataset_condition_guidance[disease_name.lower()] = {
            "keywords": [disease_name.lower()] + [item.lower() for item in symptoms],
            "title": disease_name,
            "possible_reasons": [
                f"Combined dataset records for {disease_name.lower()} often mention {', '.join(symptoms[:3]) or 'related symptoms'}.",
                "The exact cause can still vary based on age, severity, and other health conditions.",
            ],
            "medicines": [
                "Use treatment only with doctor guidance for the exact diagnosis.",
                "Dataset examples are for reference and not a personal prescription.",
            ],
            "self_care": [
                "Track symptom severity and duration.",
                "Seek medical review if symptoms are worsening or persistent.",
            ],
            "specialist": ", ".join(specializations) if specializations else "Relevant specialist",
            "emergency": [
                "Severe pain, breathing trouble, confusion, or sudden worsening",
                "Heavy bleeding, repeated vomiting, or inability to manage daily activity",
            ],
            "dataset_locations": locations,
        }


def _build_combined_faq(data_df, faq_df):
    rows = faq_df.fillna("").to_dict(orient="records")
    for disease, group in data_df.groupby("department"):
        disease_name = _safe_text(disease)
        if not disease_name:
            continue
        symptoms = _top_terms(group["symptom_text"], limit=5)
        doctors = [item for item in group["doctor_name"].dropna().astype(str).unique().tolist()[:3] if item]
        specializations = [item for item in group["specialization"].dropna().astype(str).unique().tolist()[:2] if item]
        locations = [item for item in group["location"].dropna().astype(str).unique().tolist()[:3] if item]
        avg_fee = pd.to_numeric(group["consultation_fee"], errors="coerce").fillna(0).mean()

        answer = [
            f"In the combined dataset, {disease_name.lower()} is commonly linked with {', '.join(symptoms) if symptoms else 'several related symptoms'}.",
        ]
        if specializations:
            answer.append(f"Patients usually consult {', '.join(specializations)}.")
        if doctors:
            answer.append(f"Example doctors include {', '.join(doctors)}.")
        if avg_fee > 0:
            answer.append(f"Average consultation fee in these records is about Rs. {int(round(avg_fee))}.")

        rows.append({
            "question": f"What should I know about {disease_name}?",
            "answer": " ".join(answer),
            "keywords": ", ".join([disease_name] + symptoms + specializations),
        })

    return pd.DataFrame(rows, columns=["question", "answer", "keywords"]).fillna("").drop_duplicates(subset=["question", "answer"])


def _augment_dataset(data_df):
    supplemental_rows = [
        {"age": 46, "gender": "Female", "doctor_name": "Dr. Lakshmi", "department": "Diabetes", "hospital": "City Care Clinic", "specialization": "Endocrinologist", "consultation_fee": 700, "feedback_text": "The doctor explained sugar control clearly and adjusted the treatment plan well.", "sentiment_label": "Positive", "average_rating": 4.6, "recommendation_label": 1, "location": "Chennai", "symptom_text": "frequent urination, excess thirst, tiredness", "complaint_category": "Frequent Urination", "qualification": "Endocrinologist", "experience_years": 12},
        {"age": 54, "gender": "Male", "doctor_name": "Dr. Prakash", "department": "High Blood Pressure", "hospital": "Heart Plus", "specialization": "Cardiologist", "consultation_fee": 900, "feedback_text": "Careful blood pressure monitoring and practical lifestyle advice were very helpful.", "sentiment_label": "Positive", "average_rating": 4.5, "recommendation_label": 1, "location": "Coimbatore", "symptom_text": "headache, dizziness, blurred vision", "complaint_category": "Headache", "qualification": "Cardiologist", "experience_years": 14},
        {"age": 29, "gender": "Male", "doctor_name": "Dr. Nivetha", "department": "Asthma", "hospital": "Breath Well Centre", "specialization": "Pulmonologist", "consultation_fee": 800, "feedback_text": "The inhaler technique was explained well and breathing improved with treatment.", "sentiment_label": "Positive", "average_rating": 4.7, "recommendation_label": 1, "location": "Madurai", "symptom_text": "wheezing, chest tightness, shortness of breath", "complaint_category": "Wheezing", "qualification": "Pulmonologist", "experience_years": 9},
        {"age": 38, "gender": "Female", "doctor_name": "Dr. Senthil", "department": "Sinusitis", "hospital": "ENT Care", "specialization": "ENT Specialist", "consultation_fee": 650, "feedback_text": "The doctor identified the sinus issue quickly and the treatment reduced facial pressure.", "sentiment_label": "Positive", "average_rating": 4.4, "recommendation_label": 1, "location": "Salem", "symptom_text": "blocked nose, facial pressure, headache", "complaint_category": "Blocked Nose", "qualification": "ENT Specialist", "experience_years": 11},
        {"age": 61, "gender": "Female", "doctor_name": "Dr. Arun", "department": "Arthritis", "hospital": "Ortho Mobility", "specialization": "Orthopedic", "consultation_fee": 750, "feedback_text": "The joint pain assessment was detailed and the exercise advice was useful.", "sentiment_label": "Positive", "average_rating": 4.5, "recommendation_label": 1, "location": "Trichy", "symptom_text": "joint pain, joint swelling, morning stiffness", "complaint_category": "Joint Pain", "qualification": "Orthopedic", "experience_years": 15},
        {"age": 35, "gender": "Female", "doctor_name": "Dr. Meera", "department": "Migraine", "hospital": "Neuro Health", "specialization": "Neurologist", "consultation_fee": 850, "feedback_text": "The migraine triggers were identified and medication helped reduce episodes.", "sentiment_label": "Positive", "average_rating": 4.8, "recommendation_label": 1, "location": "Chennai", "symptom_text": "severe headache, nausea, sensitivity to light", "complaint_category": "Severe Headache", "qualification": "Neurologist", "experience_years": 10},
        {"age": 42, "gender": "Male", "doctor_name": "Dr. Rajesh", "department": "Gastritis", "hospital": "Digestive Care", "specialization": "Gastroenterologist", "consultation_fee": 700, "feedback_text": "Dietary advice and antacids resolved the stomach burning quickly.", "sentiment_label": "Positive", "average_rating": 4.6, "recommendation_label": 1, "location": "Coimbatore", "symptom_text": "stomach pain, acidity, bloating", "complaint_category": "Stomach Pain", "qualification": "Gastroenterologist", "experience_years": 13},
        {"age": 50, "gender": "Female", "doctor_name": "Dr. Priya", "department": "Kidney Stones", "hospital": "Uro Clinic", "specialization": "Urologist", "consultation_fee": 900, "feedback_text": "The stone was diagnosed via scan and treatment plan was clear.", "sentiment_label": "Positive", "average_rating": 4.7, "recommendation_label": 1, "location": "Madurai", "symptom_text": "severe back pain, blood in urine, frequent urination", "complaint_category": "Severe Back Pain", "qualification": "Urologist", "experience_years": 16},
        {"age": 28, "gender": "Male", "doctor_name": "Dr. Karthik", "department": "Psoriasis", "hospital": "Skin Care Center", "specialization": "Dermatologist", "consultation_fee": 600, "feedback_text": "Creams and lifestyle tips improved the skin condition significantly.", "sentiment_label": "Positive", "average_rating": 4.5, "recommendation_label": 1, "location": "Salem", "symptom_text": "scaly skin patches, itching, redness", "complaint_category": "Scaly Skin", "qualification": "Dermatologist", "experience_years": 8},
        {"age": 55, "gender": "Male", "doctor_name": "Dr. Vijay", "department": "Cataract", "hospital": "Eye Vision", "specialization": "Ophthalmologist", "consultation_fee": 750, "feedback_text": "The cataract surgery was successful and vision improved.", "sentiment_label": "Positive", "average_rating": 4.9, "recommendation_label": 1, "location": "Trichy", "symptom_text": "blurred vision, difficulty seeing at night", "complaint_category": "Blurred Vision", "qualification": "Ophthalmologist", "experience_years": 18},
        {"age": 33, "gender": "Female", "doctor_name": "Dr. Anjali", "department": "Thyroid Disorder", "hospital": "Endo Clinic", "specialization": "Endocrinologist", "consultation_fee": 800, "feedback_text": "Thyroid tests were done and medication balanced the levels.", "sentiment_label": "Positive", "average_rating": 4.6, "recommendation_label": 1, "location": "Chennai", "symptom_text": "fatigue, weight gain, hair loss", "complaint_category": "Fatigue", "qualification": "Endocrinologist", "experience_years": 11},
        {"age": 47, "gender": "Male", "doctor_name": "Dr. Suresh", "department": "Depression", "hospital": "Mental Health Center", "specialization": "Psychiatrist", "consultation_fee": 1000, "feedback_text": "Counseling and medication helped manage the mood swings.", "sentiment_label": "Positive", "average_rating": 4.7, "recommendation_label": 1, "location": "Coimbatore", "symptom_text": "low mood, lack of interest, insomnia", "complaint_category": "Low Mood", "qualification": "Psychiatrist", "experience_years": 14},
        {"age": 40, "gender": "Female", "doctor_name": "Dr. Kavita", "department": "Anemia", "hospital": "Blood Health", "specialization": "Hematologist", "consultation_fee": 650, "feedback_text": "Iron supplements and diet changes improved energy levels.", "sentiment_label": "Positive", "average_rating": 4.4, "recommendation_label": 1, "location": "Madurai", "symptom_text": "tiredness, pale skin, shortness of breath", "complaint_category": "Tiredness", "qualification": "Hematologist", "experience_years": 12},
        {"age": 52, "gender": "Male", "doctor_name": "Dr. Ramesh", "department": "Pneumonia", "hospital": "Lung Care", "specialization": "Pulmonologist", "consultation_fee": 850, "feedback_text": "Antibiotics and rest cleared the lung infection effectively.", "sentiment_label": "Positive", "average_rating": 4.8, "recommendation_label": 1, "location": "Salem", "symptom_text": "cough with fever, chest pain, difficulty breathing", "complaint_category": "Cough with Fever", "qualification": "Pulmonologist", "experience_years": 17},
        {"age": 30, "gender": "Female", "doctor_name": "Dr. Nandini", "department": "UTI", "hospital": "Urinary Health", "specialization": "Urologist", "consultation_fee": 700, "feedback_text": "Urine tests confirmed infection and antibiotics worked quickly.", "sentiment_label": "Positive", "average_rating": 4.5, "recommendation_label": 1, "location": "Trichy", "symptom_text": "burning urination, frequent urination, lower abdominal pain", "complaint_category": "Burning Urination", "qualification": "Urologist", "experience_years": 9},
        {"age": 32, "gender": "Male", "doctor_name": "Dr. Arjun", "department": "Cavities", "hospital": "Smile Dental Clinic", "specialization": "Dentist", "consultation_fee": 600, "feedback_text": "The cavity filling was painless and the doctor explained aftercare clearly.", "sentiment_label": "Positive", "average_rating": 4.6, "recommendation_label": 1, "location": "Coimbatore", "symptom_text": "tooth pain, sensitivity", "complaint_category": "Tooth Pain", "qualification": "Dentist", "experience_years": 8},
        {"age": 41, "gender": "Female", "doctor_name": "Dr. Kavya", "department": "Gum Disease", "hospital": "Bright Dental Care", "specialization": "Dentist", "consultation_fee": 700, "feedback_text": "Bleeding gums improved after cleaning and the doctor gave good hygiene tips.", "sentiment_label": "Positive", "average_rating": 4.5, "recommendation_label": 1, "location": "Coimbatore", "symptom_text": "bleeding gums, gum swelling", "complaint_category": "Bleeding Gums", "qualification": "Dentist", "experience_years": 11},
        {"age": 27, "gender": "Male", "doctor_name": "Dr. Naveen", "department": "Tooth Abscess", "hospital": "City Dental Hospital", "specialization": "Dentist", "consultation_fee": 750, "feedback_text": "The abscess pain reduced quickly after treatment and follow-up was good.", "sentiment_label": "Positive", "average_rating": 4.7, "recommendation_label": 1, "location": "Coimbatore", "symptom_text": "swelling, pain", "complaint_category": "Tooth Abscess", "qualification": "Dentist", "experience_years": 6},
    ]
    supplemental_df = pd.DataFrame(supplemental_rows)
    combined_df = pd.concat([data_df, supplemental_df], ignore_index=True, sort=False)
    combined_df = combined_df.reset_index(drop=True)
    combined_df["doctor_id"] = combined_df.index.astype(int)
    return combined_df


def load_data():
    try:
        data_df = pd.read_excel(DATA_PATH, sheet_name=DATA_SHEET)
        print(f"[DATA] Loaded Excel dataset: {DATA_PATH}")
    except Exception:
        print(f"[DATA] Excel load failed, falling back to CSV: {FALLBACK_DATA_PATH}")
        data_df = pd.read_csv(FALLBACK_DATA_PATH)

    data_df = _normalize_dataset(data_df)
    data_df = _augment_dataset(data_df)
    data_df = _ensure_unique_doctor_names(data_df)
    faq_df = pd.read_csv(FAQ_PATH)
    faq_df = _build_combined_faq(data_df, faq_df)
    _train_nlp_model(data_df)
    _build_faq_index(faq_df)
    _prepare_dynamic_resources(data_df)
    return data_df, faq_df


def _train_nlp_model(data_df):
    global vectorizer, sentiment_model, label_encoder

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1200)
    label_encoder = LabelEncoder()

    texts = data_df["feedback_text"].fillna("").astype(str)
    labels = data_df["sentiment_label"].fillna("Neutral")

    X = vectorizer.fit_transform(texts)
    y = label_encoder.fit_transform(labels)

    sentiment_model = MultinomialNB()
    sentiment_model.fit(X, y)


def _build_faq_index(faq_df):
    global faq_vectorizer, faq_tfidf
    search_text = (
        faq_df["question"].fillna("").astype(str).str.strip() + " " +
        faq_df["keywords"].fillna("").astype(str).str.replace(",", " ", regex=False).str.strip()
    ).str.strip()
    faq_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    faq_tfidf = faq_vectorizer.fit_transform(search_text)


def _tokenize_text(text):
    return set(re.findall(r"[a-zA-Z]+", str(text).lower()))


def _clean_query_label(query_text):
    cleaned = " ".join(re.findall(r"[A-Za-z]+", str(query_text))).strip()
    return cleaned.title() if cleaned else "This condition"


def _meaningful_tokens(text):
    return {token for token in _tokenize_text(text) if token not in GENERIC_QUERY_WORDS}


def predict_sentiment(text):
    global vectorizer, sentiment_model, label_encoder
    if vectorizer is None or sentiment_model is None or label_encoder is None:
        return "Neutral"

    text_vec = vectorizer.transform([text])
    label_index = sentiment_model.predict(text_vec)[0]
    return label_encoder.inverse_transform([label_index])[0]


def _normalize_series(series):
    series = series.astype(float).fillna(0.0)
    if series.empty:
        return series
    min_val = series.min()
    max_val = series.max()
    if min_val == max_val:
        return series.apply(lambda x: 0.5)
    return (series - min_val) / (max_val - min_val)


CITY_COORDS = {
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Salem": (11.6643, 78.1460),
    "Trichy": (10.7905, 78.7047),
    "Erode": (11.3410, 77.7172),
    "Tirunelveli": (8.7139, 77.7564),
    "Vellore": (12.9165, 79.1325),
    "Thoothukudi": (8.7642, 78.1348),
    "Dindigul": (10.3670, 77.9803),
    "Cuddalore": (11.7449, 79.7656),
    "Kanchipuram": (12.8342, 79.7036),
    "Karur": (10.9590, 78.0766),
    "Nagapattinam": (10.7737, 79.8430),
    "Thanjavur": (10.7867, 79.1378),
}


def get_doctor_recommendations(data_df, part, age, fees, location, disease, symptom, specialization, doctor_name=""):
    part_key = part.strip().lower()
    specialties = PART_TO_SPECIALIZATIONS.get(part_key, [part])

    disease_aliases = {
        "Viral fever": "Fever",
        "Heart Attack": "Heart Disease",
        "Coronary Artery Disease": "Heart Disease",
        "Arrhythmia": "Heart Disease",
        "Heart Failure": "Heart Disease",
        "Pericarditis": "Heart Disease",
        "Migraine": "Migraine",
        "Fracture": "Fracture",
        "Skin Allergy": "Skin Allergy",
    }
    disease_value = disease_aliases.get(disease, disease) if disease else ""
    normalized_disease = str(disease_value).strip().lower()
    inferred_part = ""
    if normalized_disease:
        for key, part_slug in DISEASE_TO_PART.items():
            if key in normalized_disease:
                inferred_part = part_slug
                break

    resolved_part = part_key or inferred_part
    if resolved_part:
        specialties = PART_TO_SPECIALIZATIONS.get(resolved_part, [resolved_part])

    base_candidates = data_df.copy()
    if specialization:
        base_candidates = base_candidates[
            base_candidates["specialization"].str.contains(specialization, case=False, na=False)
        ]
        if resolved_part:
            part_pattern = "|".join([item for item in specialties if item])
            if part_pattern:
                base_candidates = base_candidates[
                    base_candidates["specialization"].str.contains(part_pattern, case=False, na=False)
                    | base_candidates["department"].str.contains(resolved_part, case=False, na=False)
                    | base_candidates["body_part"].str.contains(resolved_part, case=False, na=False)
                ]
            else:
                base_candidates = base_candidates[
                    base_candidates["department"].str.contains(resolved_part, case=False, na=False)
                    | base_candidates["body_part"].str.contains(resolved_part, case=False, na=False)
                ]
    elif resolved_part:
        part_pattern = "|".join([item for item in specialties if item])
        if part_pattern:
            base_candidates = base_candidates[
                base_candidates["specialization"].str.contains(part_pattern, case=False, na=False)
                | base_candidates["department"].str.contains(resolved_part, case=False, na=False)
                | base_candidates["body_part"].str.contains(resolved_part, case=False, na=False)
            ]
        else:
            base_candidates = base_candidates[
                base_candidates["department"].str.contains(resolved_part, case=False, na=False)
                | base_candidates["body_part"].str.contains(resolved_part, case=False, na=False)
            ]

    if doctor_name:
        base_candidates = base_candidates[
            base_candidates["doctor_name"].str.contains(doctor_name, case=False, na=False)
        ]

    def _apply_filters(include_disease=True, include_symptom=True, include_fees=True, include_location=True):
        candidates = base_candidates.copy()

        if include_disease and disease_value:
            disease_mask = candidates["department"].str.contains(
                disease_value, case=False, na=False
            ) | candidates["body_part"].str.contains(
                disease_value, case=False, na=False
            )
            if disease_mask.any():
                candidates = candidates[disease_mask]

        if include_symptom and symptom:
            symptom_mask = candidates["complaint_category"].str.contains(
                symptom, case=False, na=False
            ) | candidates["symptom_text"].str.contains(symptom, case=False, na=False)
            if symptom_mask.any():
                candidates = candidates[symptom_mask]

        if include_fees and fees is not None:
            candidates = candidates[candidates["consultation_fee"] <= float(fees)]

        if include_location and location:
            loc_series = candidates["location"].astype(str)
            loc_mask = loc_series.str.contains(location, case=False, na=False)
            if loc_mask.any():
                candidates = candidates[loc_mask]

        return candidates

    candidates = _apply_filters()

    if candidates.empty and fees is not None:
        candidates = _apply_filters(include_fees=False)

    if candidates.empty:
        candidates = _apply_filters(include_symptom=False)
    if candidates.empty:
        candidates = _apply_filters(include_symptom=False, include_disease=False)
    if candidates.empty:
        candidates = _apply_filters(include_symptom=False, include_disease=False, include_location=False)
    if candidates.empty:
        candidates = _apply_filters(
            include_symptom=False,
            include_disease=False,
            include_location=False,
            include_fees=False,
        )

    target_count = 6
    if 0 < len(candidates) < target_count:
        expanded = _apply_filters(include_symptom=False, include_disease=False)
        candidates = pd.concat([candidates, expanded], ignore_index=True).drop_duplicates()
    if 0 < len(candidates) < target_count:
        expanded = _apply_filters(
            include_symptom=False,
            include_disease=False,
            include_fees=False,
        )
        candidates = pd.concat([candidates, expanded], ignore_index=True).drop_duplicates()

    if candidates.empty:
        return []

    candidates["predicted_sentiment"] = candidates["feedback_text"].fillna("").apply(predict_sentiment)

    disease_match = pd.Series(0.0, index=candidates.index)
    if disease_value:
        disease_match = (
            candidates["department"].str.contains(disease_value, case=False, na=False)
            | candidates["body_part"].str.contains(disease_value, case=False, na=False)
        ).astype(float)

    symptom_match = pd.Series(0.0, index=candidates.index)
    if symptom:
        symptom_match = (
            candidates["complaint_category"].str.contains(symptom, case=False, na=False)
            | candidates["symptom_text"].str.contains(symptom, case=False, na=False)
        ).astype(float)

    sentiment_score = (
        candidates["predicted_sentiment"]
        .astype(str)
        .str.lower()
        .map({"positive": 1.0, "neutral": 0.5, "negative": 0.0})
        .fillna(0.5)
    )
    rating_score = _normalize_series(candidates["average_rating"])
    fee_score = 1 - _normalize_series(candidates["consultation_fee"])
    recommendation_label = candidates["recommendation_label"].fillna(0).astype(float)

    candidates["recommendation_score"] = (
        rating_score * 0.4
        + sentiment_score * 0.2
        + recommendation_label * 0.2
        + fee_score * 0.05
        + disease_match * 0.1
        + symptom_match * 0.05
    ).round(3)

    candidates = candidates.sort_values(
        by=["recommendation_score", "average_rating", "consultation_fee"],
        ascending=[False, False, True],
    )
    # Avoid duplicate doctor names and hospitals in results; keep the highest-ranked entry.
    candidates = candidates.drop_duplicates(subset=["doctor_name"], keep="first")
    candidates = candidates.drop_duplicates(subset=["hospital"], keep="first")
    return candidates.head(15).to_dict(orient="records")


def get_doctor_details(data_df, doctor_id):
    if "doctor_id" not in data_df.columns:
        return None
    row = data_df[data_df["doctor_id"] == doctor_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def get_map_data(data_df, location=None):
    candidates = data_df.copy()
    if location:
        candidates = candidates[candidates["location"].str.contains(location, case=False, na=False)]
    candidates = candidates.drop_duplicates(subset=["location", "doctor_name"])
    map_data = []
    for _, row in candidates.iterrows():
        city = str(row.get("location", "")).strip()
        coords = CITY_COORDS.get(city)
        if not coords:
            continue
        map_data.append({
            "doctor_id": int(row.get("doctor_id", -1)),
            "doctor": row.get("doctor_name", "Unknown"),
            "specialization": row.get("specialization", ""),
            "hospital": row.get("hospital", ""),
            "location": city,
            "lat": coords[0],
            "lon": coords[1],
            "rating": row.get("average_rating", "N/A"),
        })
    return map_data


def answer_medical_query(faq_df, query_text):
    global faq_vectorizer, faq_tfidf
    if not query_text:
        return "Please enter a question about symptoms, disease, or treatment."

    query_lower = query_text.lower()

    # Check for fever-related queries
    if "fever" in query_lower:
        fever_advice = (
            "For fever, take over-the-counter medications like paracetamol (acetaminophen) or ibuprofen. "
            "Stay hydrated, rest, and monitor your temperature. If fever exceeds 103°F (39.4°C), persists for more than 3 days, "
            "or is accompanied by severe symptoms, consult a doctor immediately.\n\n"
            "Recommended doctors: General Physician or Family Medicine specialist."
        )
        return fever_advice

    # Check if query is about doctors
    doctor_keywords = ["doctor", "specialist", "physician", "consult", "see a", "visit"]
    is_doctor_query = any(keyword in query_lower for keyword in doctor_keywords)

    query_vec = faq_vectorizer.transform([query_text])
    similarity_scores = cosine_similarity(query_vec, faq_tfidf)[0]
    query_tokens = _meaningful_tokens(query_text)
    if not query_tokens:
        query_tokens = _tokenize_text(query_text)

    best_index = 0
    best_score = -1.0
    best_similarity = 0.0
    best_keyword_overlap = 0
    for index, (_, row) in enumerate(faq_df.iterrows()):
        keyword_tokens = _meaningful_tokens(row.get("keywords", ""))
        question_tokens = _meaningful_tokens(row.get("question", ""))
        keyword_overlap = len(query_tokens & keyword_tokens)
        question_overlap = len(query_tokens & question_tokens)
        similarity_value = similarity_scores[index]
        combined_score = (
            similarity_value +
            (keyword_overlap * 0.18) +
            (question_overlap * 0.06)
        )
        if combined_score > best_score:
            best_score = combined_score
            best_index = index
            best_similarity = similarity_value
            best_keyword_overlap = keyword_overlap

    if best_score < 0.05:  # Lower threshold to always respond
        disease_hint = ", ".join(sorted(dataset_condition_guidance.keys())[:5])
        query_label = _clean_query_label(query_text)
        general_advice = (
            f"I could not find a close direct match in the combined dataset for {query_label}. "
            "However, for general health concerns, it's important to monitor symptoms and consult a healthcare professional. "
            f"Common conditions include {disease_hint}. Please provide more details or see a doctor."
        )
        if is_doctor_query and doctor_examples:
            general_advice += f" Example doctors: {', '.join(doctor_examples[:3])}."
        return general_advice

    best_answer = str(faq_df.iloc[best_index].get("answer", "")).strip()
    if not best_answer or best_answer.lower() == "nan":
        return (
            "I found a related topic, but the answer entry is incomplete. "
            "Please try a more specific symptom or consult a doctor for guidance."
        )
    if is_doctor_query and doctor_examples:
        best_answer += f"\n\nExample doctors from the combined dataset: {', '.join(doctor_examples[:3])}."

    return best_answer


def generate_medical_guidance(faq_df, query_text):
    if not query_text:
        return {
            "title": "Medical Query Assistant",
            "summary": "Please enter a disease name, symptom, or short medical question.",
            "possible_reasons": [],
            "medicines": [],
            "self_care": [],
            "specialist": "",
            "emergency": [],
            "faq_answer": "",
            "disclaimer": "This tool gives general information only and is not a medical diagnosis or prescription.",
        }

    # Correct misspelled disease names
    corrected_query = correct_disease_name(query_text)
    if corrected_query != query_text:
        query_text = corrected_query  # Use corrected query for matching

    query_lower = query_text.lower().strip()
    normalized_query = query_lower.rstrip("s")
    matched_condition = None
    combined_guidance = {}
    combined_guidance.update(CONDITION_GUIDANCE)
    combined_guidance.update(dataset_condition_guidance)

    matched_body_part = None
    if normalized_query in BODY_PART_GUIDANCE:
        matched_body_part = normalized_query
    else:
        for body_part_key, body_part_data in BODY_PART_GUIDANCE.items():
            if any(keyword in query_lower for keyword in body_part_data.get("keywords", [])):
                matched_body_part = body_part_key
                break

    if matched_body_part:
        body_part = BODY_PART_GUIDANCE[matched_body_part]
        return {
            "title": body_part["title"],
            "summary": f"Based on your query, this looks related to {matched_body_part} problems.",
            "possible_reasons": body_part["possible_reasons"],
            "medicines": body_part["medicines"],
            "self_care": body_part["self_care"],
            "specialist": body_part["specialist"],
            "emergency": body_part["emergency"],
            "faq_answer": body_part["faq_answer"],
            "disclaimer": "This tool gives general information only and is not a medical diagnosis or personalized prescription.",
        }

    for condition in combined_guidance.values():
        if any(keyword in query_lower for keyword in condition["keywords"]):
            matched_condition = condition
            break

    meaningful_query_tokens = _meaningful_tokens(query_text)
    if not matched_condition and len(meaningful_query_tokens) <= 2:
        query_label = _clean_query_label(query_text)
        disease_hint = ", ".join(sorted(dataset_condition_guidance.keys())[:5])
        faq_answer = (
            f"I could not find a close direct match in the combined dataset for {query_label}. "
            "Try asking with a disease name, major symptom, or specialist need. "
            f"Examples available in the dataset include {disease_hint}."
        )
    else:
        faq_answer = answer_medical_query(faq_df, query_text)

    if matched_condition:
        summary = f"Based on your query, this looks related to {matched_condition['title'].lower()}."
        return {
            "title": matched_condition["title"],
            "summary": summary,
            "possible_reasons": matched_condition["possible_reasons"],
            "medicines": matched_condition["medicines"],
            "self_care": matched_condition["self_care"],
            "specialist": matched_condition["specialist"],
            "emergency": matched_condition["emergency"],
            "faq_answer": faq_answer,
            "disclaimer": "Medicines listed here are common examples for general awareness only. Age, pregnancy, allergy history, current medicines, kidney or liver disease, and severity can change what is safe. Use a qualified doctor or pharmacist before taking medicine.",
        }

    return {
        "title": "General Medical Guidance",
        "summary": f"I could not confidently match {_clean_query_label(query_text)} to one specific disease, so here is general guidance.",
        "possible_reasons": [
            "The symptom may have multiple causes",
            "A doctor may need more details like age, duration, fever, pain severity, and existing illness",
        ],
        "medicines": [
            "Avoid starting antibiotics or strong medicines on your own",
            "Use only simple over-the-counter relief if you already know it is safe for you",
        ],
        "self_care": [
            "Track the symptom duration and any triggers",
            "Drink fluids, rest, and seek in-person care if symptoms worsen",
        ],
        "specialist": "General Physician",
        "emergency": [
            "Breathing trouble",
            "Severe pain",
            "Confusion, fainting, heavy bleeding, or sudden worsening",
        ],
        "faq_answer": faq_answer,
        "disclaimer": "This tool gives general information only and is not a medical diagnosis or personalized prescription.",
    }
