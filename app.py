from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask import make_response
from datetime import datetime
from pathlib import Path
import pandas as pd
from model import load_data, get_doctor_recommendations, generate_medical_guidance, get_doctor_details, get_map_data, PART_TO_SPECIALIZATIONS
import asyncio
import inspect
import re

# Import Multilingual Support
from languages import TRANSLATIONS, SUPPORTED_LANGUAGES, get_translation, get_all_translations

# Import translation for answers
try:
    from deep_translator import GoogleTranslator as DeepGoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    DeepGoogleTranslator = None
    TRANSLATOR_AVAILABLE = False

# Import Advanced SBERT for smart doctor finding
try:
    from sbert_model_advanced import (
        find_best_doctors_for_feedback,
        analyze_feedback_with_language,
        detect_language,
        load_models as load_sbert_models,
        train_sbert_model_advanced
    )
    SBERT_AVAILABLE = True
    print("[OK] SBERT Advanced Model Imported")
except ImportError:
    print("[WARNING] SBERT Advanced Model not available, using traditional matching")
    SBERT_AVAILABLE = False

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "dev-secret-key"


def _require_login():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return None


def _patients_progress_path():
    return Path(r"c:\Users\Paramasivam\Music\patients_progress.xlsx")


def _load_patients_progress():
    path = _patients_progress_path()
    if not path.exists():
        columns = [
            "enrolled_at",
            "first_name",
            "last_name",
            "email",
            "password",
            "date_of_birth",
            "country",
            "age",
        ]
        return pd.DataFrame(columns=columns)
    return pd.read_excel(path)


def _save_patients_progress(df):
    path = _patients_progress_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_excel(path, index=False)
        return None
    except PermissionError:
        fallback = path.with_name("patients_progress_pending.xlsx")
        df.to_excel(fallback, index=False)
        return (
            "patients_progress.xlsx is open. Close it and try again. "
            "Saved to patients_progress_pending.xlsx for now."
        )


def _validate_password(value):
    if len(value) < 10:
        return "Password must be at least 10 characters."
    if not re.search(r"[A-Za-z]", value):
        return "Password must include at least one letter."
    if not re.search(r"\d", value):
        return "Password must include at least one number."
    if not re.search(r"[^A-Za-z0-9]", value):
        return "Password must include at least one symbol (like @)."
    return None


def _age_from_dob(dob_text):
    try:
        dob = datetime.strptime(dob_text, "%Y-%m-%d").date()
    except ValueError:
        return None

    today = datetime.now().date()
    years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return max(years, 0)


def _append_patient_review(review_row):
    output_path = Path(r"c:\Users\Paramasivam\Music\patient_review.xlsx")
    new_df = pd.DataFrame([review_row])
    if output_path.exists():
        existing = pd.read_excel(output_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    try:
        combined.to_excel(output_path, index=False)
    except PermissionError:
        fallback = output_path.with_name("patient_review_pending.xlsx")
        combined.to_excel(fallback, index=False)


def _load_latest_patient_review(doctor_name):
    output_path = Path(r"c:\Users\Paramasivam\Music\patient_review.xlsx")
    if not output_path.exists() or not doctor_name:
        return None
    try:
        reviews = pd.read_excel(output_path)
    except Exception:
        return None

    if reviews.empty or "doctor_name" not in reviews.columns:
        return None

    filtered = reviews[reviews["doctor_name"].astype(str) == str(doctor_name)]
    if filtered.empty:
        return None
    return filtered.iloc[-1].to_dict()


def _compute_fee_range(data_df, part_key, disease, location):
    candidates = data_df.copy()
    if disease:
        candidates = candidates[candidates["department"].str.contains(disease, case=False, na=False)]
    elif part_key:
        specialties = PART_TO_SPECIALIZATIONS.get(part_key, [])
        part_pattern = "|".join([item for item in specialties if item])
        if part_pattern:
            candidates = candidates[
                candidates["specialization"].str.contains(part_pattern, case=False, na=False)
                | candidates["department"].str.contains(part_key, case=False, na=False)
            ]
    if location:
        candidates = candidates[candidates["location"].str.contains(location, case=False, na=False)]

    fees = pd.to_numeric(candidates["consultation_fee"], errors="coerce").dropna()
    fees = fees[fees > 0]
    if fees.empty:
        return None
    return {
        "min": int(fees.min()),
        "max": int(fees.max()),
    }


def _compute_fee_range_for_part_location(data_df, part_key, location):
    candidates = data_df.copy()
    if part_key:
        specialties = PART_TO_SPECIALIZATIONS.get(part_key, [])
        part_pattern = "|".join([item for item in specialties if item])
        if part_pattern:
            candidates = candidates[
                candidates["specialization"].str.contains(part_pattern, case=False, na=False)
                | candidates["department"].str.contains(part_key, case=False, na=False)
            ]
    if location:
        candidates = candidates[candidates["location"].str.contains(location, case=False, na=False)]

    fees = pd.to_numeric(candidates["consultation_fee"], errors="coerce").dropna()
    fees = fees[fees > 0]
    if fees.empty:
        return None
    return {
        "min": int(fees.min()),
        "max": int(fees.max()),
    }


def _build_review_row(feedback_text, query, results):
    row = {
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "feedback": feedback_text,
        "patient_rating": query.get("rating", ""),
        "query_disease": query.get("disease", ""),
        "query_symptom": query.get("symptom", ""),
        "query_fees": query.get("fees", ""),
        "query_location": query.get("location", ""),
        "query_language": query.get("language", ""),
    }

    if results:
        top = results[0]
        row.update({
            "doctor_name": top.get("doctor_name", ""),
            "hospital": top.get("hospital", ""),
            "specialization": top.get("specialization", ""),
            "department": top.get("department", ""),
            "location": top.get("location", ""),
            "consultation_fee": top.get("consultation_fee", ""),
            "experience_years": top.get("experience_years", ""),
            "average_rating": top.get("average_rating", ""),
            "recommendation_score": top.get("recommendation_score", ""),
            "sentiment_label": top.get("sentiment_label", ""),
            "complaint_category": top.get("complaint_category", ""),
        })
    return row

DATA_DF, FAQ_DF = load_data()
SYMPTOMS = sorted(DATA_DF["complaint_category"].dropna().astype(str).unique().tolist())
DISEASES = sorted(DATA_DF["department"].dropna().astype(str).unique().tolist())
SPECIALIZATIONS = sorted(DATA_DF["specialization"].dropna().astype(str).unique().tolist())
LOCATIONS = sorted(DATA_DF["location"].dropna().astype(str).unique().tolist())

MAJOR_PARTS = [
    {"label": "Heart", "slug": "heart"},
    {"label": "Liver", "slug": "liver"},
    {"label": "Lungs", "slug": "lungs"},
    {"label": "Brain / Neuro", "slug": "brain"},
    {"label": "Kidney", "slug": "kidney"},
]

OTHER_PARTS = [
    {"label": "Eye", "slug": "eye"},
    {"label": "Bones", "slug": "bones"},
    {"label": "ENT", "slug": "ent"},
    {"label": "Skin & Hair", "slug": "skin-hair"},
    {"label": "Teeth", "slug": "teeth"},
    {"label": "General", "slug": "general"},
]

BODY_PARTS = MAJOR_PARTS + OTHER_PARTS

PART_TO_DISEASES = {
    "heart": [
        "Coronary Artery Disease",
        "Heart Attack",
        "Arrhythmia",
        "Heart Failure",
        "Pericarditis",
    ],
    "liver": [
        "Hepatitis",
        "Fatty Liver",
        "Cirrhosis",
        "Liver Cancer",
    ],
    "lungs": [
        "Asthma",
        "Pneumonia",
        "Tuberculosis",
        "COPD",
        "Lung Cancer",
    ],
    "brain": [
        "Stroke",
        "Migraine",
        "Epilepsy",
        "Parkinson's",
        "Alzheimer's",
    ],
    "kidney": [
        "Kidney Stones",
        "Kidney Failure",
        "UTI",
        "Nephritis",
    ],
    "eye": [
        "Conjunctivitis",
        "Cataract",
        "Glaucoma",
    ],
    "bones": [
        "Fracture",
        "Arthritis",
        "Osteoporosis",
    ],
    "ent": [
        "Sinusitis",
        "Ear Infection",
        "Tonsillitis",
    ],
    "skin-hair": [
        "Acne",
        "Eczema",
        "Psoriasis",
    ],
    "teeth": [
        "Cavities",
        "Gum Disease",
        "Tooth Abscess",
    ],
    "general": [
        "Viral fever",
        "Dengue",
        "Malaria",
        "Common cold",
        "Flu",
        "Tension headache",
        "Migraine",
        "Sinus headache",
        "Food poisoning",
        "Infection",
        "Gas",
        "Ulcer",
        "Appendicitis",
    ],
}

DISEASE_TO_SYMPTOMS = {
    "Coronary Artery Disease": ["chest pain", "breathlessness"],
    "Heart Attack": ["severe chest pain", "sweating", "nausea"],
    "Arrhythmia": ["irregular heartbeat", "dizziness"],
    "Heart Failure": ["fatigue", "swelling in legs", "shortness of breath"],
    "Pericarditis": ["chest pain", "fever"],
    "Hepatitis": ["jaundice", "fatigue", "abdominal pain"],
    "Fatty Liver": ["weakness", "mild pain"],
    "Cirrhosis": ["swelling abdomen", "confusion"],
    "Liver Cancer": ["weight loss", "pain"],
    "Asthma": ["wheezing", "breathlessness"],
    "Pneumonia": ["fever", "cough", "chest pain"],
    "Tuberculosis": ["chronic cough", "weight loss"],
    "COPD": ["breathlessness", "cough"],
    "Lung Cancer": ["coughing blood", "weight loss"],
    "Stroke": ["paralysis", "speech problem"],
    "Migraine": ["severe headache", "nausea"],
    "Epilepsy": ["seizures"],
    "Parkinson's": ["tremors", "slow movement"],
    "Alzheimer's": ["memory loss"],
    "Kidney Stones": ["severe pain", "blood in urine"],
    "Kidney Failure": ["swelling", "low urine"],
    "UTI": ["burning urination"],
    "Nephritis": ["swelling", "blood in urine"],
    "Conjunctivitis": ["redness", "itching"],
    "Cataract": ["blurred vision"],
    "Glaucoma": ["eye pressure", "vision loss"],
    "Fracture": ["pain", "swelling"],
    "Arthritis": ["joint pain", "stiffness"],
    "Osteoporosis": ["weak bones"],
    "Sinusitis": ["headache", "blocked nose"],
    "Ear Infection": ["ear pain"],
    "Tonsillitis": ["throat pain", "fever"],
    "Acne": ["pimples"],
    "Eczema": ["itching", "dry skin"],
    "Psoriasis": ["red patches"],
    "Cavities": ["tooth pain"],
    "Gum Disease": ["bleeding gums"],
    "Tooth Abscess": ["swelling", "pain"],
    "Viral fever": ["high temp", "body pain"],
    "Fever": ["high temp", "body pain"],
    "Dengue": ["fever", "joint pain"],
    "Malaria": ["fever", "chills"],
    "Common cold": ["sneezing", "runny nose"],
    "Flu": ["fever", "body pain"],
    "Tension headache": ["mild pain"],
    "Sinus headache": ["face pain"],
    "Food poisoning": ["nausea", "vomiting"],
    "Infection": ["fever", "vomiting"],
    "Gas": ["bloating", "discomfort"],
    "Ulcer": ["burning pain"],
    "Appendicitis": ["severe right pain"],
}

PART_LABELS = {part["slug"]: part["label"] for part in BODY_PARTS}

@app.route("/")
def login():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login_submit():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()
    if not email or not password:
        return render_template("login.html", login_error="Email and password are required.")

    df = _load_patients_progress()
    if df.empty or "email" not in df.columns:
        return render_template("login.html", login_error="No enrollments found. Please enroll first.")

    match = df[
        (df["email"].astype(str).str.lower() == email)
        & (df["password"].astype(str) == password)
    ]
    if match.empty:
        return render_template("login.html", login_error="Invalid email or password.")
    session["logged_in"] = True
    session["user_email"] = email
    return redirect(url_for("home"))


@app.route("/enroll", methods=["GET"])
def enroll_page():
    return render_template("enroll.html")


@app.route("/enroll", methods=["POST"])
def enroll_submit():
    first_name = request.form.get("first_name", "").strip()
    last_name = request.form.get("last_name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()
    confirm_password = request.form.get("confirm_password", "").strip()
    date_of_birth = request.form.get("date_of_birth", "").strip()
    country = request.form.get("country", "").strip()
    age = request.form.get("age", "").strip()

    if not all([first_name, last_name, email, password, confirm_password, date_of_birth, country]):
        return render_template("enroll.html", enroll_error="All enroll fields are required.")
    if password != confirm_password:
        return render_template("enroll.html", enroll_error="Passwords do not match.")
    password_error = _validate_password(password)
    if password_error:
        return render_template("enroll.html", enroll_error=password_error)

    df = _load_patients_progress()
    if not df.empty and (df["email"].astype(str).str.lower() == email).any():
        return render_template("enroll.html", enroll_error="Email already enrolled. Please login.")

    computed_age = _age_from_dob(date_of_birth)
    if computed_age is None:
        return render_template("enroll.html", enroll_error="Invalid date of birth.")

    new_row = {
        "enrolled_at": datetime.now().isoformat(timespec="seconds"),
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "password": password,
        "date_of_birth": date_of_birth,
        "country": country,
        "age": computed_age,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_error = _save_patients_progress(df)
    if save_error:
        return render_template("enroll.html", enroll_error=save_error)

    return render_template("enroll.html", enroll_success="Enrollment successful. Please login.")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "GET":
        return render_template("forgot_password.html")

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()
    confirm_password = request.form.get("confirm_password", "").strip()

    if not email or not password or not confirm_password:
        return render_template("forgot_password.html", reset_error="All fields are required.")
    if password != confirm_password:
        return render_template("forgot_password.html", reset_error="Passwords do not match.")
    password_error = _validate_password(password)
    if password_error:
        return render_template("forgot_password.html", reset_error=password_error)

    df = _load_patients_progress()
    if df.empty or "email" not in df.columns:
        return render_template("forgot_password.html", reset_error="No enrollments found. Please enroll first.")

    match_idx = df.index[df["email"].astype(str).str.lower() == email]
    if match_idx.empty:
        return render_template("forgot_password.html", reset_error="Email not found. Please enroll first.")

    df.loc[match_idx, "password"] = password
    _save_patients_progress(df)
    return render_template("forgot_password.html", reset_success="Password updated. Please login.")

@app.route("/api/translations")
def api_translations():
    return jsonify(TRANSLATIONS)

@app.route("/home")
def home():
    guard = _require_login()
    if guard:
        return guard
    return render_template("home.html", major_parts=MAJOR_PARTS, other_parts=OTHER_PARTS)

@app.route("/dashboard")
def dashboard():
    guard = _require_login()
    if guard:
        return guard
    top_doctors = DATA_DF.sort_values(by=["average_rating"], ascending=False).head(15).to_dict(orient="records")
    return render_template(
        "dashboard.html",
        total_entries=len(DATA_DF),
        top_doctors=top_doctors,
        diseases=DISEASES[:10],
        symptoms=SYMPTOMS[:10],
    )

@app.route("/form/<part>")
def form(part):
    guard = _require_login()
    if guard:
        return guard
    part_key = part.strip().lower()
    part_label = PART_LABELS.get(part_key, part.capitalize())
    diseases = PART_TO_DISEASES.get(part_key, DISEASES)
    all_symptoms = sorted({
        symptom
        for disease in diseases
        for symptom in DISEASE_TO_SYMPTOMS.get(disease, [])
    })

    return render_template(
        "form.html",
        part=part_label,
        part_key=part_key,
        symptoms=all_symptoms,
        diseases=diseases,
        locations=LOCATIONS,
        disease_symptoms=DISEASE_TO_SYMPTOMS,
    )

@app.route("/search", methods=["POST"])
def search():
    guard = _require_login()
    if guard:
        return guard
    # Load SBERT models on demand before using them
    load_sbert_on_demand()
    
    part = request.form.get("part", "")
    part_key = request.form.get("part_key", "").strip().lower()
    specialization = request.form.get("specialization", "")
    disease = request.form.get("disease", "")
    symptom = request.form.get("symptom", "")
    age = request.form.get("age", "")
    fees = request.form.get("fees", "")
    location = request.form.get("location", "")
    doctor_name = request.form.get("doctor_name", "")
    feedback = request.form.get("feedback", "")
    rating = request.form.get("rating", "")
    language = request.form.get("language", "en")  # New: language support

    try:
        fees_value = float(fees) if fees else None
    except (ValueError, TypeError):
        fees_value = None

    # IMPROVED: Use SBERT-based doctor finding if feedback is provided and SBERT available
    if feedback.strip() and SBERT_AVAILABLE:
        try:
            print(f"[INFO] Using SBERT Smart Doctor Finding for feedback: {feedback[:50]}...")
            
            # Analyze feedback using SBERT
            feedback_analysis = analyze_feedback_with_language(feedback, target_lang=language)
            
            # Find doctors using semantic matching
            results = find_best_doctors_for_feedback(
                DATA_DF,
                feedback_text=feedback,
                age=age,
                fees=fees_value,
                location=location,
                top_k=10
            )

            if part_key or part:
                resolved_part = part_key or part
                specialties = PART_TO_SPECIALIZATIONS.get(resolved_part, [resolved_part])
                part_pattern = "|".join([item for item in specialties if item])
                filtered = []
                for item in results:
                    specialization_text = str(item.get("specialization", ""))
                    department_text = str(item.get("department", ""))
                    body_part_text = str(item.get("body_part", ""))
                    if part_pattern and re.search(part_pattern, specialization_text, re.IGNORECASE):
                        filtered.append(item)
                        continue
                    if re.search(resolved_part, department_text, re.IGNORECASE):
                        filtered.append(item)
                        continue
                    if re.search(resolved_part, body_part_text, re.IGNORECASE):
                        filtered.append(item)
                        continue
                if filtered:
                    results = filtered

            if feedback.strip():
                review_row = _build_review_row(
                    feedback,
                    {
                        "disease": disease,
                        "symptom": symptom,
                        "fees": fees,
                        "location": location,
                        "language": language,
                        "rating": rating,
                    },
                    results,
                )
                _append_patient_review(review_row)
            
            # Add feedback analysis to results
            feedback_info = {
                'original': feedback,
                'detected_language': feedback_analysis['detected_language'] if feedback_analysis else 'Unknown',
                'sentiment': feedback_analysis['sentiment'] if feedback_analysis else 'Neutral',
                'sentiment_confidence': feedback_analysis['sentiment_confidence'] if feedback_analysis else 0,
                'complaint': feedback_analysis['complaint'] if feedback_analysis else 'General',
                'complaint_confidence': feedback_analysis['complaint_confidence'] if feedback_analysis else 0,
            }
            
            fee_range = _compute_fee_range(DATA_DF, part_key, disease, location)
            location_part_fee_range = _compute_fee_range_for_part_location(
                DATA_DF, part_key, location
            )
            return render_template(
                "results.html",
                part=part.capitalize(),
                part_key=part_key,
                query={
                    "age": age,
                    "fees": fees,
                    "location": location,
                    "disease": disease,
                    "symptom": symptom,
                    "feedback": feedback,
                    "language": language,
                },
                fee_range=fee_range,
                location_part_fee_range=location_part_fee_range,
                results=results,
                feedback_analysis=feedback_info,
                using_sbert=True,
            )
        except Exception as e:
            print(f"[ERROR] SBERT Error: {e}, falling back to traditional search")

    # FALLBACK: Traditional doctor recommendation
    search_part = part_key or part
    results = get_doctor_recommendations(
        DATA_DF,
        search_part,
        age,
        fees_value,
        location,
        disease,
        symptom,
        "",
        doctor_name,
    )

    if feedback.strip():
        review_row = _build_review_row(
            feedback,
            {
                "disease": disease,
                "symptom": symptom,
                "fees": fees,
                "location": location,
                "language": language,
                "rating": rating,
            },
            results,
        )
        _append_patient_review(review_row)
    
    fee_range = _compute_fee_range(DATA_DF, part_key, disease, location)
    location_part_fee_range = _compute_fee_range_for_part_location(
        DATA_DF, part_key, location
    )
    return render_template(
        "results.html",
        part=part.capitalize(),
        part_key=part_key,
        query={
            "age": age,
            "fees": fees,
            "location": location,
            "disease": disease,
            "symptom": symptom,
        },
        fee_range=fee_range,
        location_part_fee_range=location_part_fee_range,
        results=results,
        using_sbert=False,
    )

def _translate_nested_value(value, target_lang, translator):
    if isinstance(value, str):
        try:
            translated = _resolve_maybe_awaitable(
                translator.translate(value, src='en', dest=target_lang)
            )
            return translated.text if hasattr(translated, 'text') else translated
        except Exception:
            return value

        return value

    if isinstance(value, list):
        return [_translate_nested_value(item, target_lang, translator) for item in value]

    if isinstance(value, tuple):
        return tuple(_translate_nested_value(item, target_lang, translator) for item in value)

    if isinstance(value, dict):
        return {
            key: _translate_nested_value(item, target_lang, translator)
            for key, item in value.items()
        }

    return value


def _resolve_maybe_awaitable(value):
    """Resolve sync/async return values from translator methods."""
    if not inspect.isawaitable(value):
        return value

    try:
        return asyncio.run(value)
    except RuntimeError:
        # Fallback when an event loop is already running.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(value)
        finally:
            loop.close()


def _extract_translation_text(translated):
    if translated is None:
        return None
    if hasattr(translated, "text"):
        return translated.text
    if isinstance(translated, str):
        return translated
    return str(translated)


def _translate_text_with_providers(text, source_lang, target_lang):
    """Translate a single text using available providers."""
    if not text or source_lang == target_lang:
        return text

    # Provider 1: deep-translator
    if TRANSLATOR_AVAILABLE and DeepGoogleTranslator is not None:
        try:
            translated_text = DeepGoogleTranslator(source=source_lang, target=target_lang).translate(text)
            if translated_text:
                return translated_text
        except Exception:
            pass

    return text


def _translate_list_with_providers(texts, source_lang, target_lang):
    if not isinstance(texts, list):
        return texts
    return [_translate_text_with_providers(text, source_lang, target_lang) for text in texts]


def _collect_translatable_strings(value, path=()):
    """Collect all non-empty strings with their paths from nested dict/list/tuple."""
    items = []

    if isinstance(value, str):
        if value.strip():
            items.append((path, value))
        return items

    if isinstance(value, list):
        for index, item in enumerate(value):
            items.extend(_collect_translatable_strings(item, path + (index,)))
        return items

    if isinstance(value, dict):
        for key, item in value.items():
            items.extend(_collect_translatable_strings(item, path + (key,)))
        return items

    return items


def _set_nested_value(container, path, new_value):
    """Set value inside nested dict/list path."""
    target = container
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = new_value


def _get_google_lang_code(language_code):
    return {
        'ta': 'ta',
        'hi': 'hi',
        'te': 'te',
        'kn': 'kn',
        'en': 'en',
    }.get(language_code, 'en')


def _fallback_localize_medical_text(text, target_lang):
    """Offline fallback for most common medical query sentence templates."""
    if not isinstance(text, str) or not text.strip() or target_lang == "en":
        return text
    cleaned_text = " ".join(text.strip().split())

    # Pattern-based templates with placeholder support
    patterns = [
        (
            r"^Based on your query, this looks related to (.+?)\.$",
            {
                "hi": "आपके प्रश्न के आधार पर, यह {x} से संबंधित लगता है।",
                "ta": "உங்கள் கேள்வியின் அடிப்படையில், இது {x} தொடர்பானதாக தெரிகிறது.",
                "te": "మీ ప్రశ్న ఆధారంగా, ఇది {x} సమస్యకు సంబంధించినట్లు కనిపిస్తోంది.",
                "kn": "ನಿಮ್ಮ ಪ್ರಶ್ನೆಯ ಆಧಾರದ ಮೇಲೆ, ಇದು {x} ಸಮಸ್ಯೆಗೆ ಸಂಬಂಧಿಸಿದೆ ಎಂದು ಕಾಣುತ್ತದೆ.",
            },
        ),
        (
            r"^Combined dataset records for (.+?) often mention (.+?)\.$",
            {
                "hi": "संयुक्त डेटा रिकॉर्ड में {x} के लिए अक्सर {y} का उल्लेख मिलता है।",
                "ta": "{x} குறித்து ஒருங்கிணைந்த தரவு பதிவுகளில் பெரும்பாலும் {y} குறிப்பிடப்படுகிறது.",
                "te": "{x} కు సంబంధించిన కలిపిన డేటాసెట్ రికార్డుల్లో తరచుగా {y} కనిపిస్తాయి.",
                "kn": "{x} ಕುರಿತು ಸಂಯುಕ್ತ ಡೇಟಾಸೆಟ್ ದಾಖಲೆಗಳಲ್ಲಿ ಸಾಮಾನ್ಯವಾಗಿ {y} ಉಲ್ಲೇಖವಾಗುತ್ತದೆ.",
            },
        ),
    ]

    for pattern, templates in patterns:
        match = re.match(pattern, cleaned_text)
        if match and target_lang in templates:
            groups = match.groups()
            formatted = templates[target_lang]
            if len(groups) >= 1:
                formatted = formatted.replace("{x}", groups[0])
            if len(groups) >= 2:
                formatted = formatted.replace("{y}", groups[1])
            return formatted

    static_map = {
        "The exact cause can still vary based on age, severity, and other health conditions.": {
            "hi": "सटीक कारण उम्र, गंभीरता और अन्य स्वास्थ्य स्थितियों के आधार पर बदल सकता है।",
            "ta": "சரியான காரணம் வயது, தீவிரம் மற்றும் பிற உடல்நிலை காரணிகளால் மாறலாம்.",
            "te": "ఖచ్చితమైన కారణం వయస్సు, తీవ్రత మరియు ఇతర ఆరోగ్య పరిస్థితులపై ఆధారపడి మారవచ్చు.",
            "kn": "ನಿಖರ ಕಾರಣವು ವಯಸ್ಸು, ತೀವ್ರತೆ ಮತ್ತು ಇತರ ಆರೋಗ್ಯ ಸ್ಥಿತಿಗಳ ಆಧಾರದ ಮೇಲೆ ಬದಲಾಗಬಹುದು.",
        },
        "Use treatment only with doctor guidance for the exact diagnosis.": {
            "hi": "सटीक निदान के लिए केवल डॉक्टर की सलाह से ही उपचार लें।",
            "ta": "துல்லியமான நோயறிதலுக்காக மருத்துவரின் வழிகாட்டுதலுடன் மட்டுமே சிகிச்சை பெறுங்கள்.",
            "te": "ఖచ్చితమైన నిర్ధారణ కోసం వైద్యుల సూచనతో మాత్రమే చికిత్సను ఉపయోగించండి.",
            "kn": "ನಿಖರ ರೋಗನಿರ್ಣಯಕ್ಕಾಗಿ ವೈದ್ಯರ ಮಾರ್ಗದರ್ಶನದೊಂದಿಗೆ ಮಾತ್ರ ಚಿಕಿತ್ಸೆ ಪಡೆಯಿರಿ.",
        },
        "Dataset examples are for reference and not a personal prescription.": {
            "hi": "डेटासेट के उदाहरण केवल संदर्भ के लिए हैं, व्यक्तिगत पर्चे के लिए नहीं।",
            "ta": "தரவுத்தள உதாரணங்கள் குறிப்புக்காக மட்டும்; இது தனிப்பட்ட மருந்து பரிந்துரை அல்ல.",
            "te": "డేటాసెట్ ఉదాహరణలు సూచన కోసం మాత్రమే; వ్యక్తిగత ప్రిస్క్రిప్షన్ కావు.",
            "kn": "ಡೇಟಾಸೆಟ್ ಉದಾಹರಣೆಗಳು ಕೇವಲ ಸೂಚನೆಗಾಗಿ; ವೈಯಕ್ತಿಕ ಔಷಧ ಪತ್ರವಲ್ಲ.",
        },
        "Track symptom severity and duration.": {
            "hi": "लक्षणों की तीव्रता और अवधि को ट्रैक करें।",
            "ta": "அறிகுறிகளின் தீவிரமும் நீடிப்பு காலமும் கண்காணிக்கவும்.",
            "te": "లక్షణాల తీవ్రత మరియు వ్యవధిని గమనించండి.",
            "kn": "ಲಕ್ಷಣಗಳ ತೀವ್ರತೆ ಮತ್ತು ಅವಧಿಯನ್ನು ಗಮನಿಸಿ.",
        },
        "Seek medical review if symptoms are worsening or persistent.": {
            "hi": "यदि लक्षण बढ़ रहे हों या लगातार बने रहें तो डॉक्टर से जांच कराएँ।",
            "ta": "அறிகுறிகள் மோசமாவதோ அல்லது நீடிப்பதோ இருந்தால் மருத்துவரை அணுகவும்.",
            "te": "లక్షణాలు పెరుగుతున్నా లేదా కొనసాగుతున్నా వైద్య సమీక్ష పొందండి.",
            "kn": "ಲಕ್ಷಣಗಳು ಹೆಚ್ಚಾದರೆ ಅಥವಾ ಮುಂದುವರಿದರೆ ವೈದ್ಯಕೀಯ ಪರಿಶೀಲನೆ ಪಡೆಯಿರಿ.",
        },
        "General Physician": {
            "hi": "सामान्य चिकित्सक",
            "ta": "பொது மருத்துவர்",
            "te": "సాధారణ వైద్యుడు",
            "kn": "ಸಾಮಾನ್ಯ ವೈದ್ಯರು",
        },
        "Fever": {
            "hi": "बुखार",
            "ta": "காய்ச்சல்",
            "te": "జ్వరం",
            "kn": "ಜ್ವರ",
        },
    }

    if cleaned_text in static_map and target_lang in static_map[cleaned_text]:
        return static_map[cleaned_text][target_lang]

    return text


def _apply_local_fallback_on_nested(value, target_lang):
    if isinstance(value, str):
        return _fallback_localize_medical_text(value, target_lang)
    if isinstance(value, list):
        return [_apply_local_fallback_on_nested(item, target_lang) for item in value]
    if isinstance(value, tuple):
        return tuple(_apply_local_fallback_on_nested(item, target_lang) for item in value)
    if isinstance(value, dict):
        return {k: _apply_local_fallback_on_nested(v, target_lang) for k, v in value.items()}
    return value


def translate_query_to_english(query_text, source_language):
    """Translate user query to English before NLP matching."""
    if not query_text or source_language == 'en':
        return query_text

    source_lang = _get_google_lang_code(source_language)
    if source_lang == 'en':
        return query_text

    try:
        if not TRANSLATOR_AVAILABLE or DeepGoogleTranslator is None:
            return query_text
        translator = DeepGoogleTranslator(source=source_lang, target='en')
        translated = translator.translate(query_text)
        return translated if translated and translated.strip() else query_text
    except Exception as e:
        print(f"[WARNING] Query translation failed: {e}")
        return query_text


def _translate_text_deep(text, target_lang):
    """Translate text from English to the target language using deep_translator."""
    if not text or target_lang == 'en':
        return text

    if not TRANSLATOR_AVAILABLE or DeepGoogleTranslator is None:
        return text

    try:
        translator = DeepGoogleTranslator(source='en', target=target_lang)
        translated = translator.translate(text)
        return translated if translated else text
    except Exception as e:
        print(f"[WARNING] Text translation failed: {e}")
        return text


# Helper function to translate medical answer to target language
def translate_answer_to_language(answer, target_language):
    """Translate the medical guidance answer to the target language."""
    if not answer or target_language == 'en':
        return answer

    target_lang = _get_google_lang_code(target_language)
    if target_lang == 'en':
        return answer

    try:
        # Recursively translate all strings in the answer dict
        def translate_recursive(obj, lang):
            if isinstance(obj, str):
                return _translate_text_deep(obj, lang)
            elif isinstance(obj, list):
                return [translate_recursive(item, lang) for item in obj]
            elif isinstance(obj, dict):
                return {k: translate_recursive(v, lang) for k, v in obj.items()}
            else:
                return obj
        
        return translate_recursive(answer, target_lang)
    except Exception as e:
        print(f"[WARNING] Translation error: {e}")
        return answer

@app.route("/query", methods=["GET", "POST"])
def query():
    answer = None
    user_query = ""
    # Prefer explicit form/URL language over cookie to avoid stale UI mismatch.
    cookie_lang = request.cookies.get("selected_language")
    form_lang = request.form.get("language")
    arg_lang = request.args.get("language")
    language = form_lang or arg_lang or cookie_lang or "en"
    supported_language_codes = {item.get("code") for item in SUPPORTED_LANGUAGES if isinstance(item, dict)}
    if language not in supported_language_codes:
        language = "en"
    
    if request.method == "POST":
        user_query = request.form.get("query", "").strip()
        processed_query = translate_query_to_english(user_query, language)
        print(f"[INFO] Query language: {language}, query: {user_query[:80]}")
        answer = generate_medical_guidance(FAQ_DF, processed_query)
        
        # Translate answer if language is not English
        if answer and language != 'en':
            answer = translate_answer_to_language(answer, language)
    
    response = make_response(
        render_template("query.html", answer=answer, user_query=user_query, selected_language=language)
    )
    response.set_cookie("selected_language", language, max_age=31536000, samesite="Lax", path="/")
    return response

@app.route("/doctor/<int:doctor_id>")
def doctor_detail(doctor_id):
    guard = _require_login()
    if guard:
        return guard
    doctor = get_doctor_details(DATA_DF, doctor_id)
    if doctor is None:
        return render_template("404.html"), 404
    latest_review = _load_latest_patient_review(doctor.get("doctor_name"))
    return render_template("doctor_detail.html", doctor=doctor, latest_review=latest_review)


@app.route("/doctor/<int:doctor_id>/review", methods=["POST"])
def submit_doctor_review(doctor_id):
    guard = _require_login()
    if guard:
        return guard
    doctor = get_doctor_details(DATA_DF, doctor_id)
    if doctor is None:
        return render_template("404.html"), 404

    feedback = request.form.get("feedback", "").strip()
    rating = request.form.get("rating", "").strip()
    if feedback:
        review_row = {
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "new_feedback": feedback,
            "patient_rating": rating,
            "doctor_id": doctor.get("doctor_id", ""),
            "doctor_name": doctor.get("doctor_name", ""),
            "hospital": doctor.get("hospital", ""),
            "specialization": doctor.get("specialization", ""),
            "department": doctor.get("department", ""),
            "location": doctor.get("location", ""),
            "consultation_fee": doctor.get("consultation_fee", ""),
            "experience_years": doctor.get("experience_years", ""),
            "average_rating": doctor.get("average_rating", ""),
            "sentiment_label": doctor.get("sentiment_label", ""),
            "complaint_category": doctor.get("complaint_category", ""),
        }
        _append_patient_review(review_row)

    latest_review = _load_latest_patient_review(doctor.get("doctor_name"))
    return render_template("doctor_detail.html", doctor=doctor, latest_review=latest_review)

@app.route("/map")
def map_view():
    guard = _require_login()
    if guard:
        return guard
    return render_template("map.html", locations=LOCATIONS)

@app.route("/map-search", methods=["POST"])
def map_search():
    guard = _require_login()
    if guard:
        return guard
    location = request.form.get("location", "")
    map_data = get_map_data(DATA_DF, location)
    return render_template("map.html", locations=LOCATIONS, selected_location=location, map_data=map_data)

# ==================== SBERT INITIALIZATION ====================
# Load SBERT models lazily (on first request) instead of startup
# This avoids long download times when starting the Flask app
SBERT_MODELS_LOADED = False

def load_sbert_on_demand():
    global SBERT_MODELS_LOADED
    if SBERT_AVAILABLE and not SBERT_MODELS_LOADED:
        try:
            if load_sbert_models("sbert_models_advanced.pkl"):
                SBERT_MODELS_LOADED = True
                print("[OK] SBERT Advanced Models loaded successfully")
        except Exception as e:
            print(f"[ERROR] Could not load SBERT models: {e}")
            print("[INFO] Run: python train_advanced.py")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
