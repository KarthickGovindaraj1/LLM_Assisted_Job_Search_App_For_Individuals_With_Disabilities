# C:/Users/karth/PycharmProjects/FlaskProject/app.py
from flask import Flask, render_template, request
import pandas as pd
from openai import OpenAI
from typing import Dict, List, Tuple
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional Gemini import (guarded so app still runs without the package)
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Constants and Global Data Caches ---
# These caches are loaded once at startup to hold non-user-specific data.
SKILLS_DATA_FILE = "ONET Excel Files 29.3\\Skills.xlsx"
UNIQUE_SKILLS: List[str] = []
TITLE_SKILL_IMPORTANCE: Dict[str, Dict[str, float]] = {}


def _find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Find the first existing column name from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns {candidates} were found in {SKILLS_DATA_FILE}.")


def load_onet_data() -> None:
    """
    Loads and caches data from the O*NET Skills Excel file.
    This populates the global caches for unique skills and job-skill importance.
    """
    global UNIQUE_SKILLS, TITLE_SKILL_IMPORTANCE
    try:
        df = pd.read_excel(SKILLS_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: The O*NET data file was not found at {SKILLS_DATA_FILE}.")
        return

    # Find the correct column names, tolerating minor variations.
    col_title = _find_column(df, ["Title", "Occupation", "Occupation Title", "Job Title"])
    col_skill = _find_column(df, ["Element Name", "Skill", "Element"])
    col_scale = _find_column(df, ["Scale ID", "Scale", "ScaleID"])
    col_value = _find_column(df, ["Data Value", "Value", "Score"])
    col_not_rel = next((c for c in ["Not Relevant", "NotRelevant"] if c in df.columns), None)

    UNIQUE_SKILLS = sorted(list(set(df[col_skill].dropna())))

    # Filter for "Importance" scores and exclude rows marked "Not Relevant".
    imp_df = df[df[col_scale].astype(str).str.upper() == 'IM'].copy()
    if col_not_rel:
        imp_df = imp_df[~imp_df[col_not_rel].astype(str).str.upper().isin(['Y', 'YES', 'TRUE', '1'])]

    imp_df = imp_df.dropna(subset=[col_title, col_skill, col_value])

    # Build the mapping: Title -> {Skill -> Normalized Importance Score}
    mapping: Dict[str, Dict[str, float]] = {}
    for _, row in imp_df.iterrows():
        title = str(row[col_title]).strip()
        skill = str(row[col_skill]).strip()
        try:
            # Normalize O*NET importance score (1-5) to a 0-1 scale.
            imp_norm = max(0.0, min(1.0, (float(row[col_value]) - 1.0) / 4.0))
            if imp_norm > 0:
                mapping.setdefault(title, {})[skill] = imp_norm
        except (ValueError, TypeError):
            continue

    TITLE_SKILL_IMPORTANCE = mapping
    print(f"Loaded and processed data for {len(UNIQUE_SKILLS)} unique skills and {len(TITLE_SKILL_IMPORTANCE)} job titles.")


def _parse_ai_severity(val) -> float:
    """Parses severity from AI model output into a float from 0.0 to 1.0."""
    try:
        f = float(val)
        return max(0.0, min(1.0, f)) if f == f else 0.0  # Clamp to [0, 1] and handle NaN
    except (ValueError, TypeError):
        s = str(val).strip().upper()
        if s in ("NONE", "NO", "NO IMPACT"): return 0.0
        if s in ("LOW", "MILD", "MINOR"): return 0.25
        if s in ("MEDIUM", "MODERATE"): return 0.5
        if s in ("HIGH", "SEVERE"): return 0.75
        if s in ("VERY HIGH", "UNSUITABLE", "TERMINAL"): return 1.0
        return 0.0


def assess_skill_impact(user_description: str, skill: str) -> Tuple[bool, float, str]:
    """
    Asks AI to assess if a user's condition terminally affects their ability to perform a skill.
    The severity score is based on inability, not discomfort.
    """
    if not user_description or not skill:
        return False, 0.0, "No description or skill provided."

    prompt = (
        "You are an expert assessor determining if a person's condition terminally affects their ability to perform a professional skill. "
        "Your assessment must focus strictly on whether the condition makes them *unable* to perform the skill, not if it causes discomfort or difficulty if the skill can still be performed.\n\n"
        f"User's condition: ```{user_description}```\n"
        f"Skill to assess: \"{skill}\"\n\n"
        "Respond in JSON with three fields:\n"
        "{\n"
        '  "impacts_skill": "YES" or "NO",\n'
        '  "severity": A float from 0.0 to 1.0, where 1.0 means the user is terminally and completely unable to perform the skill for a job. 0.0 means there is no impact on their ability at all. The score must only reflect inability, not discomfort.\n'
        '  "reason": "A one-sentence rationale for your assessment."\n'
        "}\n"
        "Rules: 'impacts_skill' must be \"YES\" if severity > 0, and \"NO\" if severity == 0. "
        "Return only the JSON object."
    )

    # Query both models for robustness
    text_g = ask_gemini(prompt)
    text_o = ask_open_ai(prompt)

    import re, json

    def _parse_response(text: str) -> Tuple[bool, float, str, bool]:
        """Helper to parse a single model's response."""
        if not text: return False, 0.0, "", False
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                severity = _parse_ai_severity(obj.get("severity", 0.0))
                impacts = severity > 0
                reason = obj.get("reason", "No reason provided.")
                msg = f"{'YES' if impacts else 'NO'} ({severity:.2f}) - {reason}"
                return impacts, severity, msg, True
            except (json.JSONDecodeError, KeyError):
                pass
        return False, 0.0, text, False # Fallback to returning raw text

    g_imp, g_sev, g_msg, g_ok = _parse_response(text_g)
    o_imp, o_sev, o_msg, o_ok = _parse_response(text_o)

    # Reconcile the two AI responses
    if g_ok and o_ok:
        if g_imp == o_imp: # Both models agree on impact
            avg_sev = (g_sev + o_sev) / 2.0
            return g_imp, avg_sev, f"Agree: {'YES' if g_imp else 'NO'} (Avg Severity: {avg_sev:.2f})\nGemini: {g_msg}\nOpenAI: {o_msg}"
        else: # Disagreement
            final_sev = max(g_sev, o_sev) # Default to higher severity
            return final_sev > 0, final_sev, f"Disagree: Defaulting to higher severity ({final_sev:.2f})\nGemini: {g_msg}\nOpenAI: {o_msg}"
    elif g_ok:
        return g_imp, g_sev, f"Gemini: {g_msg}"
    elif o_ok:
        return o_imp, o_sev, f"OpenAI: {o_msg}"
    else:
        return False, 0.0, "Neither AI model provided a usable response."


def calculate_job_scores(
    user_skill_ratings: Dict[str, int],
    skill_severities: Dict[str, float],
    top_n: int = 25
) -> List[Tuple[str, float]]:
    """
    Calculates job scores based on user ratings, weighted by O*NET importance
    and adjusted by AI-assessed disability impact.
    """
    if not TITLE_SKILL_IMPORTANCE:
        return []

    # Normalize user's 0-7 slider ratings to a 0-1 scale.
    user_ratings_norm = {skill: max(0.0, min(1.0, val / 7.0)) for skill, val in user_skill_ratings.items()}

    # Calculate impact factors (0.0 to 1.0) from severities. Factor = 1.0 - severity.
    impact_factors = {skill: 1.0 - sev for skill, sev in skill_severities.items()}

    results = []
    for title, skill_weights in TITLE_SKILL_IMPORTANCE.items():
        numerator = 0.0
        denominator = 0.0
        for skill, weight in skill_weights.items():
            user_rating = user_ratings_norm.get(skill)
            if user_rating is not None:
                # Adjust rating by the impact factor. Default to 1.0 (no impact) if not assessed.
                factor = impact_factors.get(skill, 1.0)
                adjusted_rating = user_rating * factor
                numerator += weight * adjusted_rating
                denominator += weight
        
        if denominator > 0:
            score = numerator / denominator
            results.append((title, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def ask_gemini(prompt: str) -> str | None:
    """Wrapper for calling the Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not (api_key and genai): return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash") # Using the most cost effective model available
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None


def ask_open_ai(prompt: str) -> str | None:
    """Wrapper for calling the OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: return None
    try:
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful job and skills assessor."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"} # Enforce JSON output
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return None


@app.route('/', methods=['GET', 'POST'])
def main_page():
    """
    Main web page for user input and results display.
    On GET, shows the form.
    On POST, processes inputs, runs AI assessment, and shows job scores.
    """
    if request.method == 'POST':
        user_description = request.form.get('user_description', '').strip()
        
        # Collect skill ratings from the form's sliders.
        skill_ratings = {skill: int(request.form.get(skill, 0)) for skill in UNIQUE_SKILLS}

        skill_severities: Dict[str, float] = {}
        if user_description:
            print(f"Assessing impact of description on {len(UNIQUE_SKILLS)} skills...")
            # Use a thread pool to run AI assessments in parallel for speed.
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_map = {
                    executor.submit(assess_skill_impact, user_description, skill): skill
                    for skill in UNIQUE_SKILLS
                }
                for future in as_completed(future_map):
                    skill = future_map[future]
                    try:
                        _, severity, _ = future.result()
                        if severity > 0:
                            skill_severities[skill] = severity
                    except Exception as e:
                        print(f"Error assessing skill '{skill}': {e}")
            print("Assessment complete.")

        # Calculate job scores with the collected ratings and severities.
        job_results = calculate_job_scores(skill_ratings, skill_severities)

        return render_template(
            'index.html',
            skills=UNIQUE_SKILLS,
            scores=skill_ratings,
            results=job_results,
            submitted_message="Your results are below:"
        )

    # On GET request, just show the initial page.
    return render_template('index.html', skills=UNIQUE_SKILLS, scores=None, results=None)


if __name__ == '__main__':
    print("Starting application...")
    load_onet_data()
    # To use this app, ensure you have an `index.html` file in a `templates` folder.
    # It should have a <form method="post"> containing:
    # 1. A <textarea name="user_description"></textarea>
    # 2. An <input type="range" name="{skill_name}"> for each skill.
    # 3. A <button type="submit">Submit</button>
    app.run(debug=True)

# TODO: results changing too frequently - check it out and fix, make sure gemini models are working, maybe put a bit of money in it
# TODO: drop down menu for disabilities and severity
# TODO: modify algorithm
# TODO: add more files
# TODO: ask user for interests and preferences separately
# TODO: maybe a search for specific jobs and then show them results for that job and whether its recommended or not and why
# TODO: ask the user to input their descriptions and show them their top skills and weakest skills (and all, maybe add search) and show wht affects it and how much and what they can do with a specific skill
# TODO: run tests.
# TODO: add explanations for everything

# TODO: maybe ask users a test to determine interests or conditions

# TODO (for end of project): show progress, show code, show examples, what i missed, failures, how i could improve it in the future, reflection and conclusion