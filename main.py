import urllib.parse
from fastapi import FastAPI, HTTPException,BackgroundTasks,Request
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
import os
import requests
import base64,json,re
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # DEBUG for more detail
    format='%(asctime)s [%(levelname)s] %(message)s'
)

logger = logging.getLogger(__name__)

app=FastAPI()
load_dotenv()

# Environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

def validate_secret(secret: str) -> bool:
    return secret == os.getenv("STUDENT_SECRET")

def create_repo(repo_name: str):
    logger.info(f"Creating GitHub repo: {repo_name}")
    url = "https://api.github.com/user/repos"
    payload={"name": repo_name,
             "private":False,
             "auto_init":False,
             "license_template":"mit"
            }
    if not GITHUB_TOKEN: raise Exception("Missing GITHUB_TOKEN in .env")
    headers={
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
        }
    response=requests.post(url, json=payload, headers=headers)
    if response.status_code != 201:
        raise Exception(f"Failed to create repository: {response.json()}")
    else:
        return response.json()

def enable_pages(repo_name: str):
    logger.info(f"Enabling GitHub Pages for repo: {repo_name}")
    headers={
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json" 
    }
    url = f"https://api.github.com/repos/23f3001016/{repo_name}/pages"
    payload={
        "workflow": "legacy",
        "source": {
            "branch": "main",
            "path": "/"
        }
    }
    response=requests.post(url, json=payload, headers=headers)
    if response.status_code not in [201, 204]:
        raise Exception(f"Failed to enable GitHub Pages: {response.json()}")    
    else:
        return response.json()

def push_code(repo_name: str,files: list[dict]):
    logger.info(f"Pushing {len(files)} files to repo: {repo_name}")
    headers={
        "Authorization":f"Bearer {GITHUB_TOKEN}",
        "Accept":"application/vnd.github+json"
    }
    for file in files:
        file_name=file.get("name")
        file_content=file.get("content")
        if isinstance(file_content,bytes):
            file_content=base64.b64encode(file_content).decode('utf-8')
        else:
            file_content=base64.b64encode(str(file_content).encode('utf-8')).decode('utf-8')
        payload={
            "message":f"Add {file_name}",
            "content":file_content
        }
        response=requests.put(
            f"https://api.github.com/repos/23f3001016/{repo_name}/contents/{file_name}",
            headers=headers,
            json=payload
        )
        if response.status_code not in [200,201]:
            raise Exception(f"Failed to push {file_name}: {response.json()}") 
        logger.info("Pushed files to Github successfully")

def push_updated_code(repo_name: str,files: list[dict]):
    """ Pushes updated code into an already existing repository from round 1
    """
    #Get sha of latest commit
    logger.info(f"Pushing updated code to repo: {repo_name}")
    headers={
        "Authorization":f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json" 
    }
    for file in files:
        file_name=file.get("name")
        file_content=file.get("content")
        if isinstance(file_content,bytes):
            file_content=base64.b64encode(file_content).decode('utf-8')
        else:
            file_content=base64.b64encode(str(file_content).encode('utf-8')).decode('utf-8')
        #Get sha of the file to be updated
        url = f"https://api.github.com/repos/23f3001016/{repo_name}/contents/{file_name}"
        file_sha = requests.get(url, headers=headers).json()['sha']
        payload={
            "message":f"Update {file_name}",
            "content":file_content,
            "sha":file_sha,
            "branch":"main"
        }
        response=requests.put(
            f"https://api.github.com/repos/23f3001016/{repo_name}/contents/{file_name}",
            headers=headers,
            json=payload
        )
        if response.status_code not in [200,201]:
            raise Exception(f"Failed to push {file_name}: {response.json()}")
    logger.info(f"Finished pushing updated code to repo: {repo_name}")


def extract_seed_from_checks(checks):
    for check in checks:
        if re.search(r'\$\{\w+\}', check):
            return True
    return False

def build_system_prompt():
    return """"You are an expert full-stack developer creating production-ready static web applications.

    OUTPUT FORMAT:
    Return ONLY a valid JSON array. Each element must have exactly two keys:
    - "name": the filename (string)
    - "content": the complete file content (string with \\n for newlines)

    CRITICAL REQUIREMENTS:
    1. Generate COMPLETE, WORKING code - no placeholders, no TODOs, no comments
    2. This is a STATIC site for GitHub Pages - use only HTML, CSS, and client-side JavaScript
    3. All files must be fully implemented and ready to deploy
    4. Include README.md with setup instructions and project description
    5. Default to a SINGLE HTML file with all CSS and JavaScript inline unless the task explicitly requires multiple files
    6. Only create separate JS/CSS files if the task specifically mentions them or if code is extremely large

    CRITICAL REQUIREMENTS FOR README.MD:
    -Generate a COMPLETE, HIGH-QUALITY README.md - no placeholders, no TODOs, no generic filler
    -Include a project title, concise description, and purpose
    -Include setup instructions that work for a beginner, including prerequisites, installation, and running the project locally
    -Include usage instructions, preferably with examples or screenshots (can be referenced as placeholders if images not provided)
    -Include a features section listing key functionalities or highlights of the project
    -Include a technologies/stack section clearly specifying HTML, CSS, JavaScript (or other relevant tech)
    -Include a license section or note if open-source
    -Include a contributing section describing how others can contribute (if applicable)
    -Include a contact or author section
    -Markdown formatting must be correct: use headings, code blocks, bullet points, links where relevant
    -The README.md must be self-contained, professional, and informative enough to pass automated LLM evaluation for quality and completeness
    - For the contact section include this email only 23f3001016@ds.study.iitm.ac.in

    DATA HANDLING:
    - Attachments with URLs must be fetched using fetch() API
    - For CSV: parse with native JavaScript (split by newlines and commas) or include Papa Parse from CDN
    - Perform ALL calculations client-side in the browser
    - Handle both regular URLs and data URIs correctly

    TEMPLATE VARIABLES:
    - Variables like ${seed}, ${result}, etc. are replaced with actual values BEFORE the page loads.
    - These appear in:
      * Attachment URLs: data:text/csv;base64,${seed} → will become actual base64 data
      * Title checks: "Sales Summary ${seed}" → will contain the actual seed value
      * Calculation checks: Math.abs(value - ${result}) → will be a numeric value
    - These placeholders may appear inside URLs, titles, or numeric expressions.
    - Your code must NOT try to read or access ${...} directly.
    - By the time your code runs, they already contain final runtime values.
    - Always assume resolved data (e.g., actual URLs or numbers)

    EVALUATION CHECKS:
    - The checks array contains JavaScript expressions that will be executed to verify your implementation
    - Each check MUST pass - analyze them carefully and ensure your code satisfies every condition
    - Checks run immediately after page load - do NOT use setTimeout or delays

    DOM REQUIREMENTS:
    - If checks reference specific element IDs or selectors, create those exact elements
    - If checks verify textContent, ensure the content is set synchronously on load
    - If checks look for CSS links, include the exact CDN links required

    EXTERNAL LIBRARIES:
    - Load all external libraries (Bootstrap, Papa Parse, etc.) from CDN using exact URLs specified
    - Use jsdelivr CDN for Bootstrap 5: https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css
    - All libraries must be loaded via <link> or <script> tags in HTML

    FILE STRUCTURE RULES:
    - If task mentions "single-page" or doesn't specify multiple files: create ONE index.html with everything inline
    - Only create separate files if explicitly requested or if the project is complex (e.g., "multi-page app", "separate components")
    - Always include README.md
    - Do NOT include LICENSE file unless specifically requested
    - Avoid folder structures unless explicitly required by the task

    RESPONSE FORMAT:
    [
    {"name": "index.html", "content": "<!DOCTYPE html>\\n<html>..."},
    {"name": "README.md", "content": "# Title\\n\\nDescription..."}
    ]

    Return ONLY the JSON array. No markdown, no explanations, no code fences."""

def build_user_prompt(payload):
    task = payload.get('task', 'No task specified')
    brief = payload.get('brief', 'No brief provided')
    checks = payload.get('checks', [])
    attachments = payload.get('attachments', [])
    
    has_seed = extract_seed_from_checks(checks)
    
    attachment_details = ""
    if attachments:
        attachment_details = "\n\nATTACHMENTS:\n"
        for idx, att in enumerate(attachments, 1):
            url = att.get('url', 'no URL')
            attachment_details += f"{idx}. {att.get('name', 'unnamed')}\n"
            attachment_details += f"   URL: {url}\n"
            if url.startswith('data:'):
                attachment_details += f"   This is a data URI - fetch it directly: fetch('{url}')\n"
            else:
                attachment_details += f"   This is a regular URL - fetch it with: fetch('{url}')\n"
    
    checks_detail = ""
    if checks:
        checks_detail = "\n\nEVALUATION CHECKS (all must pass):\n"
        for idx, check in enumerate(checks, 1):
            checks_detail += f"{idx}. {check}\n"
        checks_detail += "\nAnalyze each check carefully and ensure your code makes every check return true."
    
    seed_note = ""
    if has_seed:
        seed_note = "\n\nNOTE: Checks contain ${seed} or ${result} template variables. Your code must work with these actual values during evaluation."
    
    return f"""TASK: {task}

    BRIEF:
    {brief}
    {attachment_details}
    {checks_detail}
    {seed_note}

    DELIVERABLES:
    1. index.html - single HTML file with inline CSS and JavaScript (unless task requires multiple files)
    2. README.md - project documentation

    CRITICAL:
    - Implement the COMPLETE working solution
    - Use EXACT attachment URLs provided
    - Satisfy ALL evaluation checks
    - Make data available immediately (no async delays for checks)
    - Include all required external libraries from CDN
    - Inline all custom CSS and JavaScript unless explicitly told otherwise

    Generate the complete project now as a JSON array."""

def build_user_prompt_with_context(payload):
    task = payload.get('task', 'No task specified')
    brief = payload.get('brief', 'No brief provided')
    checks = payload.get('checks', [])
    attachments = payload.get('attachments', [])
    repo_name=payload["task"]
    file_context= get_files(repo_name)
    
    has_seed = extract_seed_from_checks(checks)
    
    attachment_details = ""
    if attachments:
        attachment_details = "\n\nATTACHMENTS:\n"
        for idx, att in enumerate(attachments, 1):
            url = att.get('url', 'no URL')
            attachment_details += f"{idx}. {att.get('name', 'unnamed')}\n"
            attachment_details += f"   URL: {url}\n"
            if url.startswith('data:'):
                attachment_details += f"   This is a data URI - fetch it directly: fetch('{url}')\n"
            else:
                attachment_details += f"   This is a regular URL - fetch it with: fetch('{url}')\n"
    
    checks_detail = ""
    if checks:
        checks_detail = "\n\nEVALUATION CHECKS (all must pass):\n"
        for idx, check in enumerate(checks, 1):
            checks_detail += f"{idx}. {check}\n"
        checks_detail += "\nAnalyze each check carefully and ensure your code makes every check return true."
    
    seed_note = ""
    if has_seed:
        seed_note = "\n\nNOTE: Checks contain ${seed} or ${result} template variables. Your code must work with these actual values during evaluation."
    
    return f"""TASK: {task}

    BRIEF:
    {brief}
    {attachment_details}
    {checks_detail}
    {seed_note}
    file context: {file_context}

    DELIVERABLES:
    1. index.html - single HTML file with inline CSS and JavaScript (unless task requires multiple files)
    2. README.md - project documentation

    CRITICAL:
    - Implement the COMPLETE working solution
    - Use EXACT attachment URLs provided
    - Satisfy ALL evaluation checks
    - Make data available immediately (no async delays for checks)
    - Include all required external libraries from CDN
    - Inline all custom CSS and JavaScript unless explicitly told otherwise
    - Use the existing code as context to improve and fix issues
    - Do NOT remove any existing functionality unless it directly conflicts with the new requirements
    - Follow the same file structure and naming as before unless explicitly required to change

    Generate the complete project now as a JSON array."""

def extract_json_from_response(content):
    content = content.strip()
    
    if content.startswith('```'):
        match = re.search(r'```(?:json)?\s*\n(.*?)\n```', content, re.DOTALL)
        if match:
            content = match.group(1).strip()
    
    if not content.startswith('[') and not content.startswith('{'):
        start = content.find('[')
        if start == -1:
            start = content.find('{')
        if start != -1:
            content = content[start:]
    
    if not content.endswith(']') and not content.endswith('}'):
        end = content.rfind(']')
        if end == -1:
            end = content.rfind('}')
        if end != -1:
            content = content[:end + 1]
    return content

def validate_files(files):
    required_files = {'README.md'}
    file_names = {f['name'] for f in files}
    
    missing = required_files - file_names
    if missing:
        logger.info(f"Missing recommended files: {missing}")

    has_html = any(f['name'].endswith('.html') for f in files)
    if not has_html:
        raise ValueError("No HTML file found in generated files")
    
    for f in files:
        if not f.get('content', '').strip():
            raise ValueError(f"File {f['name']} has empty content")
    
    return True

def get_files(repo_name: str):
    logger.info(f"Fetching files from repo: {repo_name}")
    username = "23f3001016"
    branch = "main"  # use 'main' or 'master'
    folder_path = ""  # "" for root folder or "subfolder"

    # GitHub API URL to list contents
    url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{folder_path}?ref={branch}"

    response = requests.get(url)
    files_list = response.json()

    # Filter only HTML files
    html_files = [f for f in files_list if f['name'].endswith('.html')]

    result = []

    for file in html_files:
        file_url = file['download_url']  # Direct raw file link
        content_response = requests.get(file_url)
        
        if content_response.status_code == 200:
            result.append({
                "name": file['name'],
                "content": content_response.text
            })
        else:
            result.append({
                "name": file['name'],
                "content": None
            })
    return result

def code_llm(payload: dict):
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(payload)
    
    logger.info("Sending request to LLM...")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    raw_content = (response.choices[0].message.content or "").strip()

    logger.info("Recevied response from LLM")
    
    if not raw_content:
        raise ValueError("LLM returned empty response")
    
    json_content = extract_json_from_response(raw_content)
    
    try:
        data = json.loads(json_content)
        logger.info("Successfully parsed JSON response")
    except json.JSONDecodeError as e:
        logger.info(f"Failed to parse JSON: {e}")
        logger.info(f"First 500 chars of attempted parse: {json_content[:500]}")
        raise ValueError(f"Invalid JSON from LLM: {e}")
    
    if isinstance(data, dict):
        if "files" in data:
            files = data["files"]
            logger.info("Extracted files from 'files' key")
        else:
            possible_arrays = [v for v in data.values() if isinstance(v, list)]
            if possible_arrays:
                files = possible_arrays[0]
                logger.info("Extracted files from dict value")
            else:
                raise ValueError("Response is a dict but contains no file array")
    elif isinstance(data, list):
        files = data
        logger.info("Response is a direct array")
    else:
        raise ValueError(f"Unexpected response type: {type(data)}")
    
    result = []
    for f in files:
        if isinstance(f, dict) and "name" in f and "content" in f:
            result.append({"name": f["name"], "content": f["content"]})
        else:
            logger.info(f"Skipping invalid file object: {f}")
    
    if not result:
        raise ValueError("No valid file objects found in response")

    logger.info(f"Extracted {len(result)} valid files")
    
    validate_files(result)
    
    return result

def modify_code_llm(payload:dict):
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt_with_context(payload)

    logger.info("Sending request to LLM with context...")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    raw_content = (response.choices[0].message.content or "").strip()

    if not raw_content:
        raise ValueError("LLM returned empty response")

    logger.info(f"Received response ({len(raw_content)} chars)")
    
    json_content = extract_json_from_response(raw_content)
    
    try:
        data = json.loads(json_content)
        logger.info("Successfully parsed JSON response")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"First 500 chars of attempted parse: {json_content[:500]}")
        raise ValueError(f"Invalid JSON from LLM: {e}")
    
    if isinstance(data, dict):
        if "files" in data:
            files = data["files"]
            logger.info("Extracted files from 'files' key")
        else:
            possible_arrays = [v for v in data.values() if isinstance(v, list)]
            if possible_arrays:
                files = possible_arrays[0]
                logger.info("Extracted files from dict value")
            else:
                raise ValueError("Response is a dict but contains no file array")
    elif isinstance(data, list):
        files = data
        logger.info("Response is a direct array")
    else:
        raise ValueError(f"Unexpected response type: {type(data)}")
    
    result = []
    for f in files:
        if isinstance(f, dict) and "name" in f and "content" in f:
            result.append({"name": f["name"], "content": f["content"]})
        else:
            logger.info(f"Skipping invalid file object: {f}")
    
    if not result:
        raise ValueError("No valid file objects found in response")

    logger.info(f"Extracted {len(result)} valid files")
    
    validate_files(result)
    
    return result

def round1(data: dict):
    logger.info(f"Starting round 1 for task: {data.get('task')}")
    files=code_llm(data)
    repo_name=data["task"]
    create_repo(repo_name)
    push_code(repo_name,files)
    enable_pages(repo_name)
    url = f"https://api.github.com/repos/23f3001016/{repo_name}/commits/main"
    sha = requests.get(url).json()['sha']
    return {"repo_name": repo_name, "commit_sha": sha}

#TODO : Convert sha fetch into a callable function
def round2(data: dict):
    logger.info(f"Starting round 2 for task: {data.get('task')}")
    modified_files=modify_code_llm(data)
    repo_name=data["task"]
    push_updated_code(repo_name,modified_files)
    url = f"https://api.github.com/repos/23f3001016/{repo_name}/commits/main"
    sha = requests.get(url).json()['sha']
    return {"repo_name": repo_name, "commit_sha": sha}

def process_task(data: dict):
    logger.info(f"Background task started for: {data.get('task')}")
    round_no = data.get("round")
    details={}
    # Run round logic only
    if round_no == 1:
        details=round1(data) 
    elif round_no == 2:
        details=round2(data) 

    # After round logic, POST results
    post_to_evaluation(details, data)

import time
import requests

def post_to_evaluation(details: dict, data: dict):
    eval_url = data.get("evaluation_url")
    if not eval_url:
        logger.warning("No evaluation_url provided in data.")
        return None

    repo_name = details.get("repo_name")
    commit_sha = details.get("commit_sha")
    callback_payload = {
        "email": data.get("email"),
        "task": data.get("task"),
        "round": data.get("round"),
        "nonce": data.get("nonce"),
        "repo_url": f"https://github.com/23f3001016/{repo_name}",
        "commit_sha": commit_sha,
        "pages_url": f"https://23f3001016.github.io/{repo_name}/"
    }

    delay = 1  # Initial retry delay (seconds)
    max_primary_attempts = 5
    primary_attempts = 0

    # Prepare fallback URL
    fallback_url = f"https://aipipe.org/proxy/{urllib.parse.quote_plus(eval_url)}"
    use_fallback = False

    while True:
        try:
            current_url = fallback_url if use_fallback else eval_url
            response = requests.post(current_url, json=callback_payload, timeout=10)

            if response.status_code == 200:
                logger.info(f"✅ Successfully posted evaluation for task {data.get('task')} to {current_url}")
                try:
                    result = response.json()
                except ValueError:
                    result = response.text
                break  # success, exit loop

            else:
                logger.error(f"❌ Received status {response.status_code} from {current_url}, retrying in {delay}s...")

        except requests.exceptions.RequestException as e:
            logger.error(f"⚠️ Error posting evaluation to {current_url}: {e}, retrying in {delay}s...")

        time.sleep(delay)
        delay *= 2  # exponential backoff

        # Count primary attempts
        if not use_fallback:
            primary_attempts += 1
            if primary_attempts >= max_primary_attempts:
                logger.warning(f"Primary eval_url failed {max_primary_attempts} times. Switching to fallback: {fallback_url}")
                use_fallback = True
                delay = 1  # reset delay for fallback

@app.post("/handle_task")
def handle_task(data: dict,background_tasks: BackgroundTasks):
    if not validate_secret(data.get("secret", "")):
        logger.info(f"Invalid secret for {data.get('email')}, task: {data.get('task')}, round: {data.get('round')}")
        raise HTTPException(status_code=401, detail="Invalid secret")
    else:
        logger.info(f"Secret validated for {data.get('email')}, task: {data.get('task')}, round: {data.get('round')}")
        background_tasks.add_task(process_task, data)
        return JSONResponse(
            content={
                "status": "ok",
                "message": f"Task {data.get('task')} accepted and being processed."
            },
            status_code=200
        )

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Server is running!",
        "endpoint": "/handle_task",
        "method": "POST"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
