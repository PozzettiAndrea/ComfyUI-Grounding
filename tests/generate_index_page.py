#!/usr/bin/env python3
"""
Generate central landing page for all test runs
"""
import os
import json
import re
from pathlib import Path
from datetime import datetime, timezone


def scan_gh_pages_runs(gh_pages_dir):
    """Scan gh-pages directory for all run directories and extract metadata"""

    gh_pages_path = Path(gh_pages_dir)
    if not gh_pages_path.exists():
        print(f"Warning: {gh_pages_dir} does not exist")
        return []

    runs = []

    # Find all directories that are numeric (run numbers)
    for item in gh_pages_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            run_number = int(item.name)

            # Count PNG images
            png_files = list(item.glob("*.png"))
            image_count = len(png_files)

            # Try to get timestamp from index.html if it exists
            index_file = item / "index.html"
            timestamp = None
            if index_file.exists():
                try:
                    content = index_file.read_text()
                    # Extract timestamp from "Generated: YYYY-MM-DD HH:MM:SS UTC"
                    match = re.search(r'Generated: ([\d\-: ]+) UTC', content)
                    if match:
                        timestamp = match.group(1)
                except Exception as e:
                    print(f"Warning: Could not read timestamp from {index_file}: {e}")

            # If no timestamp found, use directory modification time
            if not timestamp:
                mtime = item.stat().st_mtime
                timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            runs.append({
                "run_number": run_number,
                "timestamp": timestamp,
                "image_count": image_count,
                "has_gallery": index_file.exists()
            })

    # Sort by run number (descending - newest first)
    runs.sort(key=lambda x: x["run_number"], reverse=True)

    return runs


def generate_index_html(runs, output_file, repo_name="ComfyUI-Grounding"):
    """Generate the central landing page HTML"""

    total_images = sum(run["image_count"] for run in runs)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ComfyUI-Grounding Test Results</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 2rem;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            color: #58a6ff;
        }}
        .subtitle {{
            color: #8b949e;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 1.5rem;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: #58a6ff;
            display: block;
        }}
        .stat-label {{
            color: #8b949e;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        .runs-section {{
            margin-top: 3rem;
        }}
        .section-title {{
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            color: #c9d1d9;
            border-bottom: 1px solid #30363d;
            padding-bottom: 0.5rem;
        }}
        .runs-grid {{
            display: grid;
            gap: 1rem;
        }}
        .run-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 1.5rem;
            display: grid;
            grid-template-columns: auto 1fr auto auto;
            gap: 1.5rem;
            align-items: center;
            transition: all 0.2s;
            text-decoration: none;
            color: inherit;
        }}
        .run-card:hover {{
            transform: translateX(4px);
            border-color: #58a6ff;
            box-shadow: 0 4px 12px rgba(88, 166, 255, 0.1);
        }}
        .run-number {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #58a6ff;
            min-width: 80px;
        }}
        .run-info {{
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }}
        .run-timestamp {{
            color: #8b949e;
            font-size: 0.9rem;
        }}
        .run-images {{
            color: #c9d1d9;
            font-size: 0.9rem;
        }}
        .run-badge {{
            background: #238636;
            color: #fff;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        .run-badge.latest {{
            background: #1f6feb;
        }}
        .view-link {{
            color: #58a6ff;
            text-decoration: none;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border: 1px solid #58a6ff;
            border-radius: 6px;
            transition: all 0.2s;
        }}
        .view-link:hover {{
            background: #58a6ff;
            color: #0d1117;
        }}
        .empty-state {{
            text-align: center;
            padding: 4rem 2rem;
            color: #8b949e;
        }}
        .empty-state-icon {{
            font-size: 4rem;
            margin-bottom: 1rem;
        }}
        footer {{
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid #30363d;
            text-align: center;
            color: #8b949e;
            font-size: 0.9rem;
        }}
        footer a {{
            color: #58a6ff;
            text-decoration: none;
        }}
        footer a:hover {{
            text-decoration: underline;
        }}
        @media (max-width: 768px) {{
            .run-card {{
                grid-template-columns: 1fr;
                gap: 1rem;
            }}
            .run-number {{
                font-size: 1.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Test Results Gallery</h1>
        <p class="subtitle">Visual outputs from ComfyUI-Grounding automated tests</p>

        <div class="stats-grid">
            <div class="stat-card">
                <span class="stat-value">{len(runs)}</span>
                <div class="stat-label">Test Runs</div>
            </div>
            <div class="stat-card">
                <span class="stat-value">{total_images}</span>
                <div class="stat-label">Total Images</div>
            </div>
            <div class="stat-card">
                <span class="stat-value">{runs[0]['image_count'] if runs else 0}</span>
                <div class="stat-label">Latest Run Images</div>
            </div>
        </div>

        <div class="runs-section">
            <h2 class="section-title">All Test Runs</h2>
"""

    if runs:
        html += '            <div class="runs-grid">\n'

        for idx, run in enumerate(runs):
            is_latest = idx == 0
            badge = '<span class="run-badge latest">Latest</span>' if is_latest else ''

            if not run["has_gallery"]:
                continue

            html += f"""
                <a href="./{run['run_number']}/" class="run-card">
                    <div class="run-number">#{run['run_number']}</div>
                    <div class="run-info">
                        <div class="run-timestamp">📅 {run['timestamp']}</div>
                        <div class="run-images">🖼️  {run['image_count']} image{'s' if run['image_count'] != 1 else ''}</div>
                    </div>
                    {badge}
                    <div class="view-link">View Gallery →</div>
                </a>
"""

        html += '            </div>\n'
    else:
        html += """
            <div class="empty-state">
                <div class="empty-state-icon">📭</div>
                <p>No test runs found yet. Check back after the first workflow execution!</p>
            </div>
"""

    html += f"""
        </div>

        <footer>
            <p>
                Generated automatically by
                <a href="https://github.com/{repo_name}" target="_blank">ComfyUI-Grounding</a>
                CI/CD pipeline
            </p>
            <p style="margin-top: 0.5rem;">
                Last updated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
            </p>
        </footer>
    </div>
</body>
</html>
"""

    # Write the HTML file
    output_path = Path(output_file)
    output_path.write_text(html)
    print(f"✅ Generated index page at {output_path}")
    print(f"📊 {len(runs)} runs listed")
    print(f"🖼️  {total_images} total images")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generate_index_page.py <gh-pages-dir> [output-file] [repo-name]")
        print("Example: python generate_index_page.py ./gh-pages index.html ComfyUI-Grounding")
        sys.exit(1)

    gh_pages_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "index.html"
    repo_name = sys.argv[3] if len(sys.argv) > 3 else "ComfyUI-Grounding"

    print(f"🔍 Scanning {gh_pages_dir} for test runs...")
    runs = scan_gh_pages_runs(gh_pages_dir)

    print(f"📝 Generating index page...")
    generate_index_html(runs, output_file, repo_name)


if __name__ == "__main__":
    main()
