import os, sys, json
import uuid
import snowflake.connector
from textwrap import dedent
from pathlib import Path
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import complete, CompleteOptions

# Good general-purpose Cortex model
MODEL = "openai-gpt-4.1"

# Safety limit to keep prompts compact
MAX_CODE_CHARS = 40_000
# Use the notebook's active Snowflake session
session = get_active_session()

prompt = f"""Please act as a principal-level Python code reviewer. Your review must be concise, accurate, and directly actionable, as it will be posted as a GitHub Pull Request comment.

---
# CONTEXT: HOW TO REVIEW (Apply Silently)

1.  **You are reviewing a code diff, NOT a full file.** Your input shows only the lines that have been changed. Lines starting with `+` are additions, lines with `-` are removals.
2.  **Focus your review ONLY on the added or modified lines (`+` lines).** Do not comment on removed lines (`-`) unless their removal directly causes a bug in the added lines.
3.  **Infer context.** The full file context is not available. Base your review on the provided diff. Line numbers are specified in the hunk headers (e.g., `@@ -old,len +new,len @@`).
4.  **Your entire response MUST be under 65,000 characters.** Prioritize findings with `High` or `Critical` severity. If the review is extensive, omit `Low` severity findings to meet the length constraint.

# REVIEW PRIORITIES (Strict Order)
1.  Security & Correctness
2.  Reliability & Error-handling
3.  Performance & Complexity
4.  Readability & Maintainability
5.  Testability

# ELIGIBILITY CRITERIA FOR FINDINGS (ALL must be met)
-   **Evidence:** Quote the exact changed snippet (`+` lines) and cite the new line number.
-   **Severity:** Assign {Low | Medium | High | Critical}.
-   **Impact & Action:** Briefly explain the issue and provide a minimal, safe correction.
-   **Non-trivial:** Skip purely stylistic nits (e.g., import order, line length) that a linter would catch.

# HARD CONSTRAINTS (For accuracy & anti-hallucination)
-   Do NOT propose APIs that don’t exist for the imported modules.
-   Treat parameters like `db_path` as correct dependency injection; do NOT call them hardcoded.
-   NEVER suggest logging sensitive user data or internal paths. Suggest non-reversible fingerprints if context is needed.
-   Do NOT recommend removing correct type hints or docstrings.
-   If code in the diff is already correct and idiomatic, do NOT invent problems.

---
# OUTPUT FORMAT (Strict, professional, audit-ready)

Your entire response MUST be under 65,000 characters. Prioritize findings with High or Critical severity. If the review is extensive, omit Low severity findings to meet the length constraint.

## Code Review Summary
*A 2-3 sentence high-level summary. Mention the key strengths and the most critical areas for improvement across all changed files.*

---
### Detailed Findings
*A list of all material findings. If no significant issues are found, state "No significant issues found."*

**File:** `path/to/your/file.py`
-   **Severity:** {Critical | High | Medium | Low}
-   **Line:** {line_number}
-   **Function/Context:** `{function_name_if_applicable}`
-   **Finding:** {A clear, concise description of the issue, its impact, and a recommended correction.}

**(Repeat for each finding in each file)**

---
### Key Recommendations
*Provide 2-3 high-level, actionable recommendations for improving the overall quality of the codebase based on the findings. Do not repeat the findings themselves.*

---
# CODE DIFF TO REVIEW

{PY_CONTENT}

"""

response_format = {
  "type": "json",
  "schema": {
    "type": "object",
    "properties": {
      "summary": {
        "type": "string",
        "description": "A 2-3 sentence high-level summary of the code review."
      },
      "detailed_findings": {
        "type": "array",
        "description": "A list of all material findings from the code review.",
        "items": {
          "type": "object",
          "properties": {
            "file_path": {
              "type": "string",
              "description": "The full path to the file where the issue was found."
            },
            "severity": {
              "type": "string",
              "enum": ["Low", "Medium", "High", "Critical"],
              "description": "The assessed severity of the finding."
            },
            "line_number": {
              "type": "number",
              "description": "The specific line number of the issue in the new file version."
            },
            "function_context": {
              "type": "string",
              "description": "The name of the function or class where the issue is located."
            },
            "finding": {
              "type": "string",
              "description": "A clear, concise description of the issue, its impact, and a recommended correction."
            },
          },
          "required": ["file_path", "severity", "line_number", "finding"]
        }
      },
      "key_recommendations": {
        "type": "array",
        "description": "A list of high-level, actionable recommendations.",
        "items": {
          "type": "string"
        }
      }
    },
    "required": ["summary", "detailed_findings", "key_recommendations"]
  }
}


openai_response_format = {
  "type": "json",
  "schema": {
    "type": "object",
    "properties": {
      "summary": {
        "type": "string",
        "description": "A 2-3 sentence high-level summary of the code review."
      },
      "detailed_findings": {
        "type": "array",
        "description": "A list of all material findings from the code review.",
        "items": {
          "type": "object",
          "properties": {
            "file_path": {
              "type": "string",
              "description": "The full path to the file where the issue was found."
            },
            "severity": {
              "type": "string",
              "enum": ["Low", "Medium", "High", "Critical"],
              "description": "The assessed severity of the finding."
            },
            "line_number": {
              "type": "number",
              "description": "The specific line number of the issue in the new file version."
            },
            "function_context": {
              "type": "string",
              "description": "The name of the function or class where the issue is located."
            },
            "finding": {
              "type": "string",
              "description": "A clear, concise description of the issue, its impact, and a recommended correction."
            },
            "snippet": {
              "type": "string",
              "description": "The exact problematic line(s) of code from the diff."
            }
          },
          "required": [
            "file_path",
            "severity",
            "line_number",
            "function_context",
            "finding",
            "snippet"
          ],
          "additionalProperties": False
        }
      },
      "key_recommendations": {
        "type": "array",
        "description": "A list of high-level, actionable recommendations.",
        "items": {
          "type": "string"
        }
      }
    },
    "required": [
      "summary",
      "detailed_findings",
      "key_recommendations"
    ],
    "additionalProperties": False
  }
}
# Connect to Snowflake
conn = snowflake.connector.connect(
    user="manishat007",
    password="Welcome@987654321",
    account="PNQKPQT-RMB76401",
    warehouse="COMPUTE_WH",
    database="DEMO",
    schema="PUBLIC"
)

cs = conn.cursor()

def build_prompt(code_text: str) -> str:
    """Inject code into the prompt, truncating if extremely large."""
    code_text = code_text if len(code_text) <= MAX_CODE_CHARS else code_text[:MAX_CODE_CHARS]
    return dedent(PROMPT_TEMPLATE).replace("{PY_CONTENT}", code_text)

def review_with_cortex(model, code_text: str) -> str:
    """
    Calls Cortex and returns a human-readable review (starts with 'Summary:').
    Temperature=0 for deterministic, executive-style output.
    """
    messages = [
        {"role": "user", "content": build_prompt(code_text)}
    ]
    #print(build_prompt(code_text))
    out = f""" SELECT SNOWFLAKE.CORTEX.complete(
        model=model,
        prompt=messages,
        session=session,
        options=CompleteOptions(temperature=0,response_format=openai_response_format if model.startswith('openai') else response_format)
    )"""
    cs.execute(out)
    review = cs.fetchone()[0]
    return review

folder_path = _temp_diff_chunks
print(f"Processing files in '{folder_path}' and saving reviews to '{output_folder_path}'")

review_results = {} # To store results in memory if needed
for filename in os.listdir(folder_path):
    if filename.endswith(".py") or filename.endswith(".sql"):  # Process only Python files
        file_path = os.path.join(folder_path, filename)
        print(f"\n--- Reviewing file: {filename} ---")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_to_review = f.read()

            if not code_to_review.strip():
                print(f"  Skipping empty file: {filename}")
                review_text = "No code found in file, skipping review."
            else:
                review_text = review_with_cortex(model, code_to_review, session)
                print(f"  Review received for {filename}.")

            # Save the individual response
            output_filename = f"{Path(filename).stem}_review.md" # or .txt, .json
            output_file_path = os.path.join(output_folder_path, output_filename)
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                outfile.write(review_text)
            print(f"  Review saved to: {output_file_path}")

            review_results[filename] = review_text
          except Exception as e:
            print(f"  Error processing file '{filename}': {e}", file=sys.stderr)
            # You might want to save an error message for this file as well
            error_output_filename = f"{Path(filename).stem}_error.txt"
            error_output_file_path = os.path.join(output_folder_path, error_output_filename)
            with open(error_output_file_path, 'w', encoding='utf-8') as outfile:
                outfile.write(f"ERROR processing file: {e}\n")
                outfile.write(f"Original file path: {file_path}\n")

print(f"\nFinished processing all files in '{folder_path}'.")
# Save review as a GitHub Action output if needed
# delimiter = str(uuid.uuid4())

# max_github_comment_length = 60000
# short_review = review[:max_github_comment_length]
# if len(review) > max_github_comment_length:
#     short_review += "\n\n---\n**This review is truncated. The full review is available as a workflow artifact.**"
# with open('full_review.md', 'w') as f:
#     f.write(review)

# # The rest: write short_review to GITHUB_OUTPUT as before
# with open(os.environ['GITHUB_OUTPUT'], 'a') as gh_out:
#     gh_out.write(f'review<<{delimiter}\n')
#     gh_out.write(f'{review}\n')
#     gh_out.write(f'{delimiter}\n')
