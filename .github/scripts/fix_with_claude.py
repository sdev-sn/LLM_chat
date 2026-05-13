import anthropic
import os
import sys

error_title = os.environ.get("ERROR_TITLE", "Unknown error")
error_culprit = os.environ.get("ERROR_CULPRIT", "")
sentry_url = os.environ.get("SENTRY_URL", "")

with open("app.py") as f:
    app_code = f.read()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=4096,
    system=(
        "You are an expert Python developer fixing bugs in a Streamlit app. "
        "Return ONLY the complete fixed Python file content. "
        "No explanations, no markdown code blocks, no commentary."
    ),
    messages=[{
        "role": "user",
        "content": (
            f"Fix this error in the Streamlit app:\n\n"
            f"Error: {error_title}\n"
            f"Location: {error_culprit}\n\n"
            f"Current app.py:\n{app_code}"
        ),
    }],
)

fixed_code = response.content[0].text.strip()

# Strip markdown fences if the model added them anyway
if fixed_code.startswith("```python"):
    fixed_code = fixed_code[len("```python"):].lstrip("\n")
if fixed_code.startswith("```"):
    fixed_code = fixed_code[3:].lstrip("\n")
if fixed_code.endswith("```"):
    fixed_code = fixed_code[:-3].rstrip("\n")

with open("app.py", "w") as f:
    f.write(fixed_code + "\n")

print(f"Fix applied for: {error_title}")
