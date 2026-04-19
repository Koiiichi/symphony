"""Brain Agent Instructions - Centralized prompt templates.

Provides instructions for code generation and fixing based on sensory feedback.
"""

from typing import Dict, Any
from agents.sensory_contract import SensoryReport


def get_generation_instructions(
    project_root: str,
    goal: str,
    stack: Dict[str, Any]
) -> str:
    """Generate instructions for initial code generation.
    
    Args:
        project_root: Absolute path to project directory
        goal: Natural language goal from user
        stack: Detected stack information
        
    Returns:
        Formatted instructions for Brain agent
    """
    
    if stack.get("has_content"):
        # Project exists, make targeted updates
        frameworks = ", ".join(stack.get("frameworks", ["unknown"]))
        
        return f"""
You are working on an existing project at: {project_root}

DETECTED STACK:
- Frontend: {stack.get('frontend', 'unknown')}
- Backend: {stack.get('backend', 'unknown')}
- Frameworks: {frameworks}

USER GOAL:
{goal}

TASK:
1. First, use list_project_files() to see the current structure
2. Use read_existing_code() to examine key files
3. Make MINIMAL, TARGETED changes to achieve the goal
4. Preserve existing functionality
5. Follow the existing code style and patterns

GUIDELINES:
- DO NOT rewrite working code
- DO NOT change the overall architecture
- DO NOT create standalone one-off test scripts (e.g. test_*.py at project root) unless explicitly requested
- DO make targeted improvements that align with the goal
- DO ensure all changes are consistent with existing stack
- Use write_code() only for files that need changes

Start by examining the current project structure.
"""
    
    else:
        # Empty project, scaffold from scratch
        return f"""
You are starting a NEW project at: {project_root}

USER GOAL:
{goal}

TASK: Create a complete, functional web application that achieves this goal.

REQUIRED STRUCTURE:
1. frontend/
   - index.html (complete HTML with embedded CSS and JavaScript)
   - package.json (if using Node.js features)

2. backend/
   - app.py (Flask backend with CORS enabled)
   - requirements.txt (Python dependencies)

FRONTEND REQUIREMENTS:
- Responsive design (desktop and mobile)
- Modern, clean aesthetics with good spacing
- Semantic HTML with proper accessibility
- Working interactive elements (forms, buttons)
- If the goal mentions a contact form, it must:
  * Capture user input (name, email, message)
  * Submit to /api/contact endpoint
  * Display success/error messages
  * Handle validation

BACKEND REQUIREMENTS:
- Flask app serving frontend as static files
- CORS enabled for all routes
- POST endpoint at /api/contact that:
  * Accepts JSON with name, email, message
  * Validates required fields
  * Returns JSON response with success status
- Proper error handling

VISUAL REQUIREMENTS:
- Consistent spacing scale (8px, 16px, 24px, 32px, 48px)
- Clear visual hierarchy
- High contrast text (WCAG AA compliant)
- Professional color palette
- Smooth interactions and transitions

PROCESS:
1. Use write_code() to create frontend/index.html
2. Use write_code() to create frontend/package.json
3. Use write_code() to create backend/app.py
4. Use write_code() to create backend/requirements.txt

IMPORTANT:
- Do not create extra ad-hoc scripts outside this app structure unless explicitly asked.

Start generating the complete application now.
"""


def get_fix_instructions(
    project_root: str,
    report: SensoryReport,
    goal: str
) -> str:
    """Generate instructions for fixing issues from sensory report.
    
    Args:
        project_root: Absolute path to project directory
        report: Sensory agent report with test results
        goal: Original user goal
        
    Returns:
        Formatted fix instructions for Brain agent
    """
    
    failing_gates = report.get_failing_gates()
    
    if not failing_gates:
        return f"""
Project at {project_root} has passed all quality gates!

Current scores:
- Alignment: {report.alignment_score:.2f}
- Spacing: {report.spacing_score:.2f}
- Contrast: {report.contrast_score:.2f}
- Contact form: {'Working' if report.interaction.contact_submitted else 'Not applicable'}

No fixes needed. You may add polish or enhancements if desired.
"""
    
    fix_details = report.get_fix_instructions()
    
    return f"""
You are fixing issues in the project at: {project_root}

ORIGINAL GOAL:
{goal}

QUALITY GATE RESULTS:
{_format_gate_results(report)}

FAILING GATES:
{chr(10).join(f"- {gate}" for gate in failing_gates)}

{fix_details}

PROCESS:
1. Use read_existing_code() to examine the files that need fixes
2. Identify the root cause of each failing gate
3. Make TARGETED fixes to address specific issues
4. Use write_code() to update only the files that need changes
5. Ensure fixes don't break existing functionality

IMPORTANT:
- Focus on the specific issues identified in the failing gates
- Make minimal changes that directly address the problems
- Test your mental model: will this fix improve the failing metric?
- Preserve all working functionality

Start by reading the files that likely need fixes.
"""


def _format_gate_results(report: SensoryReport) -> str:
    """Format quality gate results for display.
    
    Args:
        report: Sensory report
        
    Returns:
        Formatted results string
    """
    lines = [
        f"- Alignment: {report.alignment_score:.2f} (threshold: 0.90)",
        f"- Spacing: {report.spacing_score:.2f} (threshold: 0.90)",
        f"- Contrast: {report.contrast_score:.2f} (threshold: 0.75)",
        f"- Visible sections: {', '.join(report.visible_sections) or 'none'}",
        f"- Contact form: {'Working' if report.interaction.contact_submitted else 'Failed'}",
    ]
    
    if report.interaction.details:
        lines.append(f"  Details: {report.interaction.details}")
    
    if report.a11y.violations > 0:
        lines.append(f"- Accessibility: {report.a11y.violations} violations (threshold: 5)")
        for issue in report.a11y.top_issues[:3]:
            lines.append(f"  • {issue}")
    
    if not report.playwright.passed:
        lines.append(f"- Tests: {len(report.playwright.failed_tests)} failed")
        for test in report.playwright.failed_tests[:3]:
            lines.append(f"  • {test}")
    
    return "\n".join(lines)


def get_scaffold_template(app_type: str = "portfolio") -> Dict[str, str]:
    """Get template files for scaffolding a new project.
    
    Args:
        app_type: Type of application (portfolio, dashboard, ecommerce, etc.)
        
    Returns:
        Dictionary mapping file paths to content templates
    """
    
    templates = {
        "portfolio": {
            "frontend/index.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: system-ui, -apple-system, sans-serif;
            line-height: 1.6;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        header {
            padding: 24px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        .hero {
            padding: 80px 0;
            text-align: center;
        }
        .hero h1 {
            font-size: 48px;
            margin-bottom: 16px;
        }
        .projects {
            padding: 80px 0;
        }
        .project-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 24px;
        }
        .project-card {
            padding: 24px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        .contact {
            padding: 80px 0;
        }
        .contact form {
            max-width: 600px;
            margin: 0 auto;
        }
        .form-group {
            margin-bottom: 16px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
        }
        .form-group input,
        .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: inherit;
        }
        button {
            background: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <nav>
                <a href="#hero">Home</a>
                <a href="#projects">Projects</a>
                <a href="#contact">Contact</a>
            </nav>
        </div>
    </header>

    <section id="hero" class="hero">
        <div class="container">
            <h1>Welcome to My Portfolio</h1>
            <p>Building beautiful web experiences</p>
        </div>
    </section>

    <section id="projects" class="projects">
        <div class="container">
            <h2>Projects</h2>
            <div class="project-grid">
                <div class="project-card">
                    <h3>Project One</h3>
                    <p>Description of project one</p>
                </div>
                <div class="project-card">
                    <h3>Project Two</h3>
                    <p>Description of project two</p>
                </div>
            </div>
        </div>
    </section>

    <section id="contact" class="contact">
        <div class="container">
            <h2>Contact</h2>
            <form id="contact-form">
                <div class="form-group">
                    <label for="name">Name</label>
                    <input type="text" id="name" name="name" required>
                </div>
                <div class="form-group">
                    <label for="email">Email</label>
                    <input type="email" id="email" name="email" required>
                </div>
                <div class="form-group">
                    <label for="message">Message</label>
                    <textarea id="message" name="message" rows="5" required></textarea>
                </div>
                <button type="submit">Send Message</button>
                <div id="form-status"></div>
            </form>
        </div>
    </section>

    <script>
        document.getElementById('contact-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const statusEl = document.getElementById('form-status');
            
            const data = {
                name: document.getElementById('name').value,
                email: document.getElementById('email').value,
                message: document.getElementById('message').value
            };
            
            try {
                const response = await fetch('/api/contact', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (response.ok && result.success) {
                    statusEl.textContent = result.message || 'Message sent successfully!';
                    statusEl.style.color = 'green';
                    e.target.reset();
                } else {
                    statusEl.textContent = result.error || 'Failed to send message';
                    statusEl.style.color = 'red';
                }
            } catch (error) {
                statusEl.textContent = 'Network error. Please try again.';
                statusEl.style.color = 'red';
            }
        });
    </script>
</body>
</html>""",
            
            "backend/app.py": """from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/contact', methods=['POST'])
def contact():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    message = data.get('message', '').strip()
    
    if not (name and email and message):
        return jsonify({
            'success': False,
            'error': 'Missing required fields'
        }), 400
    
    # In production, send email or save to database
    return jsonify({
        'success': True,
        'message': 'Thank you for your message!'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
""",
            
            "backend/requirements.txt": """Flask>=2.0.0
Flask-Cors>=3.0.0
"""
        }
    }
    
    return templates.get(app_type, templates["portfolio"])
