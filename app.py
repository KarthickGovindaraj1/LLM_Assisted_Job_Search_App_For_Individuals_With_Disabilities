# C:/Users/karth/PycharmProjects/FlaskProject/app.py
from flask import Flask, render_template, request
import pandas as pd
from openai import OpenAI
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

# Temporary per-skill impact flags collected from user description (aligned with _UNIQUE_SKILLS order)
# Values are strings "YES" or "NO"; may be empty if not collected in this session.
userDescImpactsYN: List[str] = []
# New: per-skill severity aligned with _UNIQUE_SKILLS order.
# Values are floats in [0, 1], where:
# 0 = no impact on job suitability; 1 = completely unsuitable for the job.
userDescSeverity: List[float] = []


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


def _build_skill_impact_factor_map(skills: List[str]) -> Dict[str, float]:
    """Return a mapping skill -> factor adjusted by user-described impact severity.

    Severity is on a continuous scale from 0.0 to 1.0:
    - 0.0 = no impact on job suitability
    - 1.0 = completely unsuitable for the job

    Factor applied to user rating r is (1 - severity).

    Fallback: if per-skill severities are not available, use YES/NO flags with
    IMPACT_YES_FACTOR (env; default 0.6) for YES and 1.0 for NO.
    """
    # Fallback factor for YES if severities aren't available
    try:
        raw = os.getenv("IMPACT_YES_FACTOR", "0.6")
        fallback_yes_factor = float(raw)
        if not (0.0 <= fallback_yes_factor <= 1.0):
            fallback_yes_factor = 0.6
    except Exception:
        fallback_yes_factor = 0.6

    if not skills:
        return {}

    # Access globals; if absent, return empty map (no adjustments)
    try:
        global userDescImpactsYN, userDescSeverity
    except Exception:
        return {}

    factor_map: Dict[str, float] = {}

    # Prefer severities if available
    if userDescSeverity:
        L = min(len(skills), len(userDescSeverity))
        for i in range(L):
            sk = skills[i]
            try:
                sev = float(userDescSeverity[i])
            except Exception:
                sev = 0.0
            # Clamp severity to [0,1] and compute factor = 1 - severity
            if sev != sev:  # NaN guard
                sev = 0.0
            sev = max(0.0, min(1.0, sev))
            factor_map[sk] = 1.0 - sev
        return factor_map

    # Fall back to YES/NO flags if provided
    if not userDescImpactsYN:
        return {}

    L = min(len(skills), len(userDescImpactsYN))
    for i in range(L):
        sk = skills[i]
        flag = str(userDescImpactsYN[i]).strip().upper()
        factor_map[sk] = (fallback_yes_factor if flag == "YES" else 1.0)
    return factor_map


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
    - Applies a per-skill factor derived from user-described impact severity if available: factor = 1 - severity,
      where severity is a float from 0.0 to 1.0 denoting job suitability impact (0=no impact, 1=unsuitable).
      Falls back to YES/NO impacts with IMPACT_YES_FACTOR (env; default 0.6) when severities are unavailable.
    - Score per title = sum(w * (r * factor)) / sum(w), where r is user rating normalized to 0–1.
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

    # Build impact factor map from global YES/NO flags; if unavailable, returns empty and factors default to 1.0
    skills_for_map = _UNIQUE_SKILLS if _UNIQUE_SKILLS else list(user_norm.keys())
    impact_factor_map = _build_skill_impact_factor_map(skills_for_map)

    results: List[Tuple[str, float]] = []
    for title, skill_weights in _TITLE_SKILL_IMPORTANCE.items():
        num = 0.0
        den = 0.0
        for skill, w in skill_weights.items():
            r = user_norm.get(skill)
            if r is None:
                continue
            factor = impact_factor_map.get(skill, 1.0)
            r_adj = r * factor
            num += w * r_adj
            den += w
        if den > 0:
            score = num / den  # 0–1
            results.append((title, score))

    # Sort by score desc and return top N
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def ask_gemini(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-pro")
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if not text:
            # Fallback attempt to extract text if SDK exposes candidates
            try:
                text = resp.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
            except Exception:
                text = ""
        print("Gemini:", text)
        return text
    except Exception:
        return None

def ask_open_ai(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are helping the user decide if various conditions affect a skill/ability/job."},
                {"role": "user", "content": prompt}
            ]
        )
        print("OpenAI:", completion.choices[0].message.content)
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        return None


# ask_gemini_skill_impact you could kind of clone for another LLM model.
def ask_gemini_skill_impact(user_desc: str, skill: str) -> Tuple[bool, float, str]:
    """Ask both Gemini and OpenAI whether the user's description affects the specified skill and how severely.

    Severity is on a continuous scale from 0.0 to 1.0:
    - 0.0 = no impact on job suitability
    - 0.5 = moderate impact on suitability for job
    - 1.0 = completely unsuitable for the job

    Returns (impacts_skill, severity_0to1, message) where:
    - impacts_skill is a boolean for YES/NO impact
    - severity_0to1 is a float in [0.0, 1.0]
    - message includes a concise summary or error/explainer.
    """
    # Basic validation
    if not user_desc or not skill:
        return False, 0.0, "No description or skill provided; cannot assess."

    def _parse_severity(val) -> float:
        """Parse severity from model output to a float between 0.0 and 1.0."""
        try:
            f = float(val)
            if f != f:  # NaN guard
                return 0.0
            return max(0.0, min(1.0, f))  # Clamp to [0, 1]
        except (ValueError, TypeError):
            s = str(val).strip().upper()
            if s in ("NONE", "NO", "NO IMPACT", "NOT IMPACTED"):
                return 0.0
            if s in ("LOW", "L", "MILD", "MINOR", "SLIGHT"):
                return 0.25
            if s in ("MEDIUM", "MID", "M", "MODERATE"):
                return 0.5
            if s in ("HIGH", "H", "SEVERE"):
                return 0.75
            if s in ("VERY HIGH", "VERY_HIGH", "VERYHIGH", "VH", "UNSUITABLE", "COMPLETELY UNSUITABLE"):
                return 1.0
            return 0.0

    try:
        prompt = (
            "You are assessing whether a user's description affects their ability to perform a specific skill.\n\n"
            f"User description: ```{user_desc}```\n"
            f"Skill to assess: \"{skill}\"\n\n"
            "Respond strictly in JSON with three fields only: \n"
            "{\n  \"impacts_skill\": \"YES\" or \"NO\",\n  \"severity\": a float from 0.0 to 1.0 (0.0 = no impact on job suitability, 1.0 = completely unsuitable for the job),\n  \"reason\": \"a one-sentence brief rationale\"\n}\n"
            "Rules: If severity > 0 then impacts_skill must be \"YES\"; if severity == 0 then impacts_skill must be \"NO\".\n"
            "Only return the JSON object."
        )

        # Query both models if available
        text_g = ask_gemini(prompt)
        text_o = ask_open_ai(prompt)

        import re, json  # local import to keep globals minimal

        def _parse_response(text: str) -> Tuple[bool, float, str, bool]:
            """Parse a model response into (impacts, severity, message, ok). ok=False if no usable content."""
            if not text:
                return False, 0.0, "", False
            impacts: bool = False
            severity: float = 0.0
            msg: str = text.strip() if isinstance(text, str) else ""

            match = re.search(r"\{.*\}", text or "", re.DOTALL)
            if match:
                try:
                    obj = json.loads(match.group(0))
                    val = str(obj.get("impacts_skill", "")).strip().upper()
                    impacts = val.startswith("Y")
                    severity = _parse_severity(obj.get("severity", 0.0))
                    # Enforce consistency with severity scale: impacts iff severity > 0
                    impacts = (severity > 0)
                    reason = obj.get("reason") or ""
                    msg = f"{'YES' if impacts else 'NO'} ({severity:.2f}) - {reason}".strip()
                    return impacts, severity, msg, True
                except Exception:
                    pass

            upper = (text or "").upper()
            if "YES" in upper and "NO" not in upper:
                impacts = True
            elif "NO" in upper and "YES" not in upper:
                impacts = False
            else:
                # ambiguous
                return False, 0.0, msg or "", False
            try:
                nums = [float(n) for n in re.findall(r"0?\.\d+|1\.0+|1\b", text or "")]
                if nums:
                    # Use the parsed number, clamped to [0,1]
                    severity = _parse_severity(max(nums))
                else:
                    if any(k in upper for k in ["UNSUITABLE", "COMPLETELY UNSUITABLE", "VERY HIGH", "VH"]):
                        severity = 1.0
                    elif ("HIGH" in upper) or ("SEVERE" in upper):
                        severity = 0.75
                    elif ("MEDIUM" in upper) or ("MODERATE" in upper) or ("MILD" in upper):
                        severity = 0.5
                    elif ("LOW" in upper) or ("NO IMPACT" in upper) or ("NONE" in upper):
                        severity = 0.25
                    else:
                        severity = 0.0
            except Exception:
                severity = 0.0
            # Enforce impacts based on severity
            impacts = (severity > 0)
            return impacts, severity, msg or "", True

        g_ok = False
        o_ok = False
        g_imp = False
        o_imp = False
        g_sev = 0.0
        o_sev = 0.0
        g_msg = ""
        o_msg = ""

        if text_g is not None:
            g_imp, g_sev, g_msg, g_ok = _parse_response(text_g)
        if text_o is not None:
            o_imp, o_sev, o_msg, o_ok = _parse_response(text_o)

        # No responses
        if not g_ok and not o_ok:
            backend_info = []
            if text_g is None:
                backend_info.append("Gemini: unavailable")
            if text_o is None:
                backend_info.append("OpenAI: unavailable")
            hint = "; ".join(backend_info) or "Both models returned empty responses."
            return False, 0.0, f"No usable response from either model. {hint}"

        # Only one usable
        if g_ok and not o_ok:
            return g_imp, g_sev, f"Gemini: {g_msg or ('YES' if g_imp else 'NO')}"
        if o_ok and not g_ok:
            return o_imp, o_sev, f"OpenAI: {o_msg or ('YES' if o_imp else 'NO')}"

        # Both usable -> reconcile
        if g_imp == o_imp:
            avg_sev = max(0.0, min(1.0, (g_sev + o_sev) / 2.0))
            agree = "YES" if g_imp else "NO"
            return g_imp, avg_sev, f"Agree: {agree} (avg severity {avg_sev:.2f}). Gemini: {g_msg}; OpenAI: {o_msg}"
        else:
            # If they disagree on impact, default to higher severity
            final_sev = max(g_sev, o_sev)
            final_imp = final_sev > 0
            return final_imp, final_sev, f"Disagree: Gemini says {'YES' if g_imp else 'NO'} ({g_sev:.2f}), OpenAI says {'YES' if o_imp else 'NO'} ({o_sev:.2f}). Defaulting to higher severity."

    except Exception as e:
        print(e)
        return False, 0.0, f"Error assessing via AI: {e}"

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
    """Prompt for a description, then ask LLMs for EVERY skill; store severity and YES/NO per skill.

    Severity is on a continuous scale from 0.0 to 1.0 (0=no impact, 1=unsuitable).
    Parallelizes LLM requests to speed up processing while preserving the original order of skills.
    Configure concurrency with env var LLM_MAX_WORKERS (default 5). Falls back to sequential if needed.
    """
    global userDesc, userDescImpactsYN, userDescSeverity
    try:
        userDesc = input("Enter any disability or description you have: ").strip()
    except Exception:
        userDesc = ""

    # Local import to avoid broad file changes
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
    except Exception:
        ThreadPoolExecutor = None  # type: ignore
        as_completed = None  # type: ignore

    # Ensure skills are loaded
    if not _UNIQUE_SKILLS or not _TITLE_SKILL_IMPORTANCE:
        _load_skills_cache()

    skills = _UNIQUE_SKILLS if _UNIQUE_SKILLS else get_unique_elements(_SKILLS_FILE, "Element Name")

    # Temporary list storing ONLY "YES" or "NO" per skill in the same order as `skills`
    # Try parallel execution if the concurrency tools are available
    def _sequential():
        seq_impacts = []
        seq_severity = []
        for skill in skills:
            try:
                impacts, sev, _msg = ask_gemini_skill_impact(userDesc, skill)
                impacts = (sev > 0)
                seq_impacts.append("YES" if impacts else "NO")
                seq_severity.append(sev)
            except Exception:
                seq_impacts.append("NO")
                seq_severity.append(0.0)
            print(skill, "is done")
        return seq_impacts, seq_severity

    if not skills:
        userDescImpactsYN = []
        userDescSeverity = []
        print("No skills loaded; nothing to assess.")
        print(userDescImpactsYN)
        return

    if ThreadPoolExecutor is None or as_completed is None:
        userDescImpactsYN, userDescSeverity = _sequential()
    else:
        # Parallel path
        max_workers_str = os.getenv("LLM_MAX_WORKERS", "5")
        try:
            max_workers = max(1, int(float(max_workers_str)))
        except Exception:
            max_workers = 5

        userDescImpactsYN = ["NO"] * len(skills)
        userDescSeverity = [0.0] * len(skills)
        print(f"Submitting {len(skills)} skills to Gemini with max_workers={max_workers}...")
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(ask_gemini_skill_impact, userDesc, skill): (idx, skill)
                              for idx, skill in enumerate(skills)}
                done = 0
                for future in as_completed(future_map):
                    idx, skill = future_map[future]
                    try:
                        impacts, sev, _msg = future.result()
                        # Enforce impacts based on severity and store both
                        try:
                            sev = float(sev)
                        except Exception:
                            sev = 0.0
                        sev = max(0.0, min(1.0, sev))
                        impacts = (sev > 0)
                        userDescSeverity[idx] = sev
                        userDescImpactsYN[idx] = "YES" if impacts else "NO"
                    except Exception:
                        userDescImpactsYN[idx] = "NO"
                        userDescSeverity[idx] = 0.0
                    done += 1
                    if done % 10 == 0 or done == len(skills):
                        print(f"Processed {done}/{len(skills)} skills...")
        except Exception as e:
            print(f"Parallel execution error: {e}. Falling back to sequential processing.")
            userDescImpactsYN, userDescSeverity = _sequential()

    print(f"Collected AI YES/NO impacts for {len(skills)} skills into userDescImpactsYN.")
    print(userDescImpactsYN)  # parallelize llm requests for speed
    try:
        print("Collected per-skill severities (0=no impact, 1=unsuitable):")
        print(userDescSeverity)
    except Exception:
        pass

# also use dropdown for list and severity
# search for a specific job with inputs and seeing compatiility and any issues involved
# job has to be assessed directly with users description, possibly in secondary weghting category
# if it retuns above if there is high severity and no, check again, default to high if no change, also default to higher severity if there is both yes, if both no, reutnrn, if theres low and no, default t low, if mild and no affect, leave it mild


if __name__ == '__main__':
    user_Disabilities()
    app.run(debug=True)