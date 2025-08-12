# C:/Users/karth/PycharmProjects/FlaskProject/app.py
from flask import Flask, render_template, request
import pandas as pd
from typing import Dict, List, Tuple
import os

# Optional Gemini import (guarded so app still runs without the package)
try:
    import google.generativeai as genai
except Exception:
    genai = None

app = Flask(__name__)

# Global cache for skills data
_SKILLS_FILE = "ONET Excel Files 29.3\\Skills.xlsx"
_UNIQUE_SKILLS: List[str] = []
_TITLE_SKILL_IMPORTANCE: Dict[str, Dict[str, float]] = {}


def _safe_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return the first existing column name from candidates; raise if none present."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns {candidates} were found in the Skills.xlsx file.")


def _load_skills_cache() -> None:
    """Load Skills.xlsx and populate global caches for unique skills and title-skill importance mapping."""
    global _UNIQUE_SKILLS, _TITLE_SKILL_IMPORTANCE
    try:
        df = pd.read_excel(_SKILLS_FILE)
    except FileNotFoundError:
        print(f"Error: The file at {_SKILLS_FILE} was not found.")
        _UNIQUE_SKILLS = []
        _TITLE_SKILL_IMPORTANCE = {}
        return

    # Resolve column names (be tolerant to slight naming differences)
    col_title = _safe_col(df, ["Title", "Occupation", "Occupation Title", "Job Title"])  # Titles per O*NET
    col_skill = _safe_col(df, ["Element Name", "Skill", "Element"])
    col_scale = _safe_col(df, ["Scale ID", "Scale", "ScaleID"])  # Importance vs Level
    col_value = _safe_col(df, ["Data Value", "Value", "Score"])  # Numeric value
    # Not Relevant column may be named exactly or similar
    col_not_rel = None
    for c in ["Not Relevant", "NotRelevant", "Not_Relevant", "Not relevant", "Not- Relevant", "NotRel"]:
        if c in df.columns:
            col_not_rel = c
            break

    # Unique skills across all rows
    _UNIQUE_SKILLS = sorted(list(set(df[col_skill].dropna())))

    # Filter to Importance scale and relevant rows
    imp_df = df[df[col_scale].astype(str).str.upper() == 'IM'].copy()
    if col_not_rel is not None:
        # Exclude rows marked as Not Relevant (commonly 'Y')
        imp_df = imp_df[~imp_df[col_not_rel].astype(str).str.upper().isin(['Y', 'YES', 'TRUE', '1'])]

    # Drop rows with missing essentials
    imp_df = imp_df.dropna(subset=[col_title, col_skill, col_value])

    # Build mapping: title -> {skill -> normalized_importance}
    mapping: Dict[str, Dict[str, float]] = {}
    for _, row in imp_df.iterrows():
        title = str(row[col_title]).strip()
        skill = str(row[col_skill]).strip()
        try:
            imp = float(row[col_value])
        except Exception:
            continue
        # Importance in O*NET is 1–5; normalize to 0–1. Clamp just in case.
        imp_norm = max(0.0, min(1.0, (imp - 1.0) / 4.0))
        if imp_norm <= 0:
            continue
        mapping.setdefault(title, {})[skill] = imp_norm

    _TITLE_SKILL_IMPORTANCE = mapping


def get_unique_elements(file_path: str, column_name: str) -> list:
    """Reads an Excel file and returns a sorted list of unique elements from a column."""
    try:
        file_frame = pd.read_excel(file_path)
        item_set = set(file_frame[column_name].dropna())
        return sorted(list(item_set))
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return []
    except KeyError:
        print(f"Error: The column '{column_name}' was not found in the file.")
        return []


def _compute_job_scores(user_scores: Dict[str, int], top_n: int = 25) -> List[Tuple[str, float]]:
    """Compute and return top-N job titles scored by weighted match against user skill ratings.

    - user_scores: raw slider ratings per skill (0–7 scale).
    - Uses importance weights per title (normalized 0–1) from Skills.xlsx (Scale ID == 'IM'), ignoring Not Relevant.
    - Score per title = sum(w * r) / sum(w), where r is user rating normalized to 0–1.
    """
    if not _TITLE_SKILL_IMPORTANCE:
        _load_skills_cache()
    if not _TITLE_SKILL_IMPORTANCE:
        return []

    # Normalize user scores to 0–1
    user_norm: Dict[str, float] = {}
    for skill, raw in user_scores.items():
        try:
            val = float(raw)
        except Exception:
            continue
        # Slider is 0–7 (inclusive). Normalize; clamp between 0 and 1.
        user_norm[skill] = max(0.0, min(1.0, val / 7.0))

    results: List[Tuple[str, float]] = []
    for title, skill_weights in _TITLE_SKILL_IMPORTANCE.items():
        num = 0.0
        den = 0.0
        for skill, w in skill_weights.items():
            r = user_norm.get(skill)
            if r is None:
                continue
            num += w * r
            den += w
        if den > 0:
            score = num / den  # 0–1
            results.append((title, score))

    # Sort by score desc and return top N
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def ask_gemini_skill_impact(user_desc: str, skill: str) -> Tuple[bool, str]:
    """Ask Gemini whether the user's description affects the specified skill.

    Returns (impacts_skill, message) where impacts_skill is a boolean and message
    includes a concise YES/NO plus brief rationale or an error/explainer.
    """
    # Basic validation
    if not user_desc or not skill:
        return False, "No description or skill provided; cannot assess."

    # Ensure SDK is available
    if genai is None:
        return (
            False,
            "Gemini client not available. Install 'google-generativeai' and set GEMINI_API_KEY to enable AI assessment.",
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False, "GEMINI_API_KEY environment variable is not set; cannot contact Gemini."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are assessing whether a user's description affects their ability to perform a specific skill.\n\n"
            f"User description: ```{user_desc}```\n"
            f"Skill to assess: \"{skill}\"\n\n"
            "Respond strictly in JSON with two fields: \n"
            "{\n  \"impacts_skill\": \"YES\" or \"NO\",\n  \"reason\": \"a one-sentence brief rationale\"\n}\n"
            "Only return JSON."
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if not text:
            # Fallback attempt to extract text if SDK exposes candidates
            try:
                text = resp.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
            except Exception:
                text = ""

        # Try to parse JSON from response
        import re, json  # local import to keep globals minimal

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                val = str(obj.get("impacts_skill", "")).strip().upper()
                impacts = val.startswith("Y")
                reason = obj.get("reason") or ""
                msg = f"{'YES' if impacts else 'NO'} - {reason}".strip()
                return impacts, msg
            except Exception:
                pass

        # Fallback: simple heuristic
        upper = text.upper()
        if "YES" in upper and "NO" not in upper:
            impacts = True
        elif "NO" in upper and "YES" not in upper:
            impacts = False
        else:
            impacts = False
        return impacts, (text.strip() or "No response from Gemini.")

    except Exception as e:
        return False, f"Error calling Gemini: {e}"


@app.route('/', methods=['GET', 'POST'])
def skill_sliders():
    """
    Renders a page with sliders for each unique skill from the O*NET Skills file.
    Handles form submission to capture skill scores and compute matching job titles.
    """
    # Ensure caches are loaded once
    if not _UNIQUE_SKILLS or not _TITLE_SKILL_IMPORTANCE:
        _load_skills_cache()

    unique_skills = _UNIQUE_SKILLS if _UNIQUE_SKILLS else get_unique_elements(_SKILLS_FILE, "Element Name")

    results: List[Tuple[str, float]] = []
    scores_dict: Dict[str, int] = {}

    if request.method == 'POST':
        # Collect submitted slider values
        for skill in unique_skills:
            score = request.form.get(skill)
            if score is not None and score != "":
                try:
                    scores_dict[skill] = int(float(score))
                except Exception:
                    continue

        # Compute ranked job titles based on importance-weighted match
        results = _compute_job_scores(scores_dict, top_n=25)

        return render_template(
            'index.html',
            skills=unique_skills,
            submitted_message="Your scores have been submitted!",
            scores=scores_dict,
            results=results
        )

    # GET request
    return render_template('index.html', skills=unique_skills, scores=None, results=None)

def user_Disabilities():
    """Prompt for a description and a specific skill, then ask Gemini about impact."""
    global userDesc, userSkillQuery, userDescImpact
    try:
        userDesc = input("Enter any disability or description you have: ").strip()
    except Exception:
        userDesc = ""
    try:
        userSkillQuery = input("Enter the specific skill to assess impact on (e.g., 'Active Listening'): ").strip()
    except Exception:
        userSkillQuery = ""

    impacts, message = ask_gemini_skill_impact(userDesc, userSkillQuery)
    userDescImpact = {"impacts": impacts, "message": message}
    print(f"Gemini assessment for skill '{userSkillQuery}': {message}")

# also use dropdown for list and severity

if __name__ == '__main__':
    user_Disabilities()
    app.run(debug=True)