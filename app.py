# C:/Users/karth/PycharmProjects/FlaskProject/app.py
from flask import Flask, render_template, request
import pandas

app = Flask(__name__)


def get_unique_elements(file_path: str, column_name: str) -> list:
    """Reads an Excel file and returns a sorted list of unique elements from a column."""
    try:
        file_frame = pandas.read_excel(file_path)
        item_set = set(file_frame[column_name].dropna())
        return sorted(list(item_set))
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return []
    except KeyError:
        print(f"Error: The column '{column_name}' was not found in the file.")
        return []


@app.route('/', methods=['GET', 'POST'])
def skill_sliders():
    """
    Renders a page with sliders for each unique skill from the O*NET Skills file.
    Handles form submission to capture skill scores.
    """
    skills_file = "ONET Excel Files 29.3/Skills.xlsx"
    skills_column = "Element Name"
    unique_skills = get_unique_elements(skills_file, skills_column)

    # This block executes when the form is submitted via POST
    if request.method == 'POST':
        skill_scores = []
        for skill in unique_skills:
            # request.form.get() retrieves the value for each skill from the submitted form
            score = request.form.get(skill)
            if score:
                # Add the skill and its integer score as a tuple to our list
                skill_scores.append((skill, int(score)))

        # --- This is where the scores are stored in a list ---
        # For this example, we'll print them to the console.
        # In a real application, you would save this to a database or process it further.
        print("--- User Submitted Scores ---")
        print(skill_scores)
        print("---------------------------")

        # Re-render the page with a success message and the submitted scores
        # to keep the sliders at their submitted positions.
        return render_template('index.html',
                               skills=unique_skills,
                               submitted_message="Your scores have been submitted!",
                               scores=dict(skill_scores))

    # This block executes for a GET request (the initial page load)
    return render_template('index.html', skills=unique_skills, scores=None)


if __name__ == '__main__':
    app.run(debug=True)