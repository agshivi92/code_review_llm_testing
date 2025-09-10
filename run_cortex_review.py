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
MAX_CHARS_FOR_FINAL_SUMMARY_FILE = 65000
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
PROMPT_TEMPLATE_CONSOLIDATED_SUMMARY = f"""
You are an AI assistant tasked with consolidating multiple individual code review feedbacks. Your goal is to produce a single, concise summary of all provided reviews.

Highlight the most important errors and potential issues, *especially those that explicitly mention line numbers or code blocks*. If multiple reviews point to similar issues (e.g., "inconsistent error handling" across several files), consolidate them into a single recommendation.

The output should be professional, actionable, and suitable for a high-level overview.

---
# OUTPUT FORMAT (Strict, professional, concise)

Your entire response MUST be under {MAX_CHARS_FOR_FINAL_SUMMARY_FILE} characters. Prioritize findings with High or Critical severity.

## Consolidated Code Review Summary
*A 2-3 sentence high-level summary of the entire code change across all files.*

---
### Key Findings Across Files
*A consolidated list of the most critical and recurring issues from the individual reviews. For each finding, mention the files involved if applicable and highlight line numbers where available. If no significant issues are found, state "No significant issues found across files."*

**Example Finding:**
-   **Severity:** High
-   **Issue:** Inconsistent error handling (e.g., `File: file1.py, Line: 45`; `File: file2.py, Line: 120`). Implement a standardized error logging and exception handling strategy.

---
### Overall Recommendations
*Provide 2-3 high-level, actionable recommendations for improving the overall quality of the codebase, consolidating insights from all individual reviews.*

---
# INDIVIDUAL CODE REVIEWS TO CONSOLIDATE

{{ALL_REVIEWS_CONTENT}}

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

def count_tokens(text: str) -> int:
    """Counts tokens using the configured tiktoken encoder."""
    return len(TOKENIZER_ENCODING.encode(text))

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """Truncates text to ensure it does not exceed max_tokens."""
    tokens = TOKENIZER_ENCODING.encode(text)
    if len(tokens) > max_tokens:
        print(f"Warning: Text exceeds {max_tokens} tokens ({len(tokens)}). Truncating.", file=sys.stderr)
        return TOKENIZER_ENCODING.decode(tokens[:max_tokens])
    return text


def build_prompt(code_text: str) -> str:
    """Inject code into the prompt, truncating if extremely large."""
    #code_text = code_text if len(code_text) <= MAX_CODE_CHARS else code_text[:MAX_CODE_CHARS]
    return dedent(PROMPT_TEMPLATE).replace("{PY_CONTENT}", code_text)

def build_prompt_for_consolidated_summary(all_reviews_content: str) -> str:
    """Inject combined reviews into the summary prompt, truncating if too large."""
    #all_reviews_content_truncated = truncate_by_tokens(all_reviews_content, MAX_TOKENS_FOR_SUMMARY_INPUT)
    return dedent(PROMPT_TEMPLATE_CONSOLIDATED_SUMMARY).replace("{{ALL_REVIEWS_CONTENT}}", all_reviews_content_truncated)
  
def review_with_cortex(model, code_text: str) -> str:
    """
    Calls Cortex and returns a human-readable review (starts with 'Summary:').
    Temperature=0 for deterministic, executive-style output.
    """
    messages = [
        {"role": "user", "content": build_prompt(code_text)}
    ]
    #print(build_prompt(code_text))
    try:
      review = complete(
          model=model,
          prompt=messages,
          session=session,
          options=CompleteOptions(temperature=0,response_format=openai_response_format if model.startswith('openai') else response_format)
      )
      return review
    except Exception as e:
        print(f"Error calling Cortex complete for model '{model}': {e}", file=sys.stderr)
        return f"ERROR: Could not get response from Cortex. Reason: {e}"

# --- Main Script Logic ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_cortex_review.py  <input_directory_path> <output_directory_path>", file=sys.stderr)
        sys.exit(1)

    folder_path = sys.argv[1]
    output_folder_path = sys.argv[2]
    consolidated_summary_filename = "consolidated_code_review_summary.md"
    print(f"Processing files in '{folder_path}' and saving reviews to '{output_folder_path}'")
    if os.path.exists(output_folder_path):
        import shutil
        shutil.rmtree(output_folder_path)
        print(f"Cleaned up previous '{output_folder_path}' directory.")
      
    os.makedirs(output_folder_path, exist_ok=True)

    all_individual_reviews_text = []  # List To store results in memory if needed
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
                  review_prompt_messages = [
                          {"role": "user", "content": build_prompt_for_individual_review(code_to_review)}
                      ]
                  review_text = review_with_cortex(model, review_prompt_messages, session)
                  print(f"  Review received for {filename}.")
                  
                all_individual_reviews_text.append(f"--- Review for {filename} ---\n{review_text}\n")

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
                error_review_text = f"--- Review for {filename} ---\nERROR: Could not generate review for this file. Reason: {e}\n"
                all_individual_reviews_text.append(error_review_text)

    print(f"\nFinished processing all files in '{folder_path}'.")
    
    consolidated_summary_text =''
    if not all_individual_reviews_text:
      print("No individual reviews were generated to summarize. Skipping summarization.")
      sys.exit(0) # Exit if no reviews to summarize

    combined_reviews_content = "\n".join(all_individual_reviews_text) 
    summary_prompt_messages = [
            {"role": "user", "content": build_prompt_for_consolidated_summary(combined_reviews_content)}
        ]
    consolidated_summary_text = call_cortex_complete(MODEL, summary_prompt_messages, session)
    if len(consolidated_summary_text) > MAX_CHARS_FOR_FINAL_SUMMARY_FILE:
        print(f"Warning: Final consolidated summary exceeds {MAX_CHARS_FOR_FINAL_SUMMARY_FILE} characters. Truncating for file save.", file=sys.stderr)
        consolidated_summary_text = consolidated_summary_text[:MAX_CHARS_FOR_FINAL_SUMMARY_FILE]

    print(f"  Consolidated summary received (first 500 chars): {consolidated_summary_text[:500].strip()}...")

    # Save the consolidated summary to a file
    consolidated_summary_file_path = os.path.join(output_folder_path, consolidated_summary_filename)
    try:
        with open(consolidated_summary_file_path, 'w', encoding='utf-8') as outfile:
            outfile.write(consolidated_summary_text)
        print(f"\n--- Consolidated summary saved to: {consolidated_summary_file_path} ---")
    except Exception as e:
        print(f"ERROR: Could not save consolidated summary to file: {e}", file=sys.stderr)

    print("\n--- Code review process completed. ---")
    # Save review as a GitHub Action output if needed
    delimiter = str(uuid.uuid4())

    # max_github_comment_length = 60000
    # short_review = review[:max_github_comment_length]
    # if len(review) > max_github_comment_length:
    #     short_review += "\n\n---\n**This review is truncated. The full review is available as a workflow artifact.**"
    # with open('full_review.md', 'w') as f:
    #     f.write(review)

    # # The rest: write short_review to GITHUB_OUTPUT as before
    with open(os.environ['GITHUB_OUTPUT'], 'a') as gh_out:
        gh_out.write(f'Final Review<<{delimiter}\n')
        gh_out.write(f'{consolidated_summary_text}\n')
        gh_out.write(f'{delimiter}\n')
