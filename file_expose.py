from flask import Flask, send_from_directory, abort, request
from flask_cors import CORS  # Added import
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

BASE_DIR = "/ist-nas/users/palakonk/singularity_logs/plots/fit_man_1_frame_fast"

def safe_join(base, *paths):
    # Prevent directory traversal attacks.
    final_path = os.path.abspath(os.path.join(base, *paths))
    if os.path.commonpath([final_path, base]) != os.path.abspath(base):
        abort(404)
    return final_path

@app.route('/', defaults={'req_path': ''})
@app.route('/<path:req_path>')
def dir_listing(req_path):
    # Compute the absolute path
    abs_path = safe_join(BASE_DIR, req_path)

    if os.path.isfile(abs_path):
        # Serve the file if it's a file.
        directory = os.path.dirname(abs_path)
        filename = os.path.basename(abs_path)
        return send_from_directory(directory, filename)
    elif os.path.isdir(abs_path):
        # Show directory contents
        try:
            files = os.listdir(abs_path)
        except OSError:
            abort(404)

        files_links = []
        # Parent directory link if not at the base dir.
        if os.path.abspath(abs_path) != os.path.abspath(BASE_DIR):
            parent = os.path.relpath(os.path.join(abs_path, '..'), BASE_DIR)
            files_links.append(f"<li><a href='/{parent}'>[..]</a></li>")

        for file in sorted(files):
            file_path = os.path.join(req_path, file)
            files_links.append(f"<li><a href='/{file_path}'>{file}</a></li>")
        return "<h1>Index of /{}</h1><ul>{}</ul>".format(req_path, "\n".join(files_links))
    else:
        # If path doesn't exist, return 404.
        abort(404)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)