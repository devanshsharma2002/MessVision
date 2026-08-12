import os
import shutil
from flask import Flask, render_template, url_for

app = Flask(__name__)

# Disable Flask static file caching
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Source folders
SOURCE_FOLDERS = {
    'college_non_mess': r'C:\Users\aammu\Documents\Final year project\facerecognition\detected_faces\college_non_mess',
    'outsiders': r'C:\Users\aammu\Documents\Final year project\facerecognition\detected_faces\outsiders'
}

def sync_images(source_folder, static_subfolder):
    """Copy images from source to static folder"""
    static_path = os.path.join('static', static_subfolder)
    os.makedirs(static_path, exist_ok=True)
    
    # Copy all images
    for file in os.listdir(source_folder):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(source_folder, file)
            dst = os.path.join(static_path, file)
            if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                shutil.copy2(src, dst)

@app.after_request
def add_header(response):
    """Add headers to prevent caching"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def display_college_non_mess():
    sync_images(SOURCE_FOLDERS['college_non_mess'], 'college_non_mess')
    
    static_folder = 'static/college_non_mess'
    images = [f for f in os.listdir(static_folder) 
              if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    images.sort(key=lambda x: os.path.getmtime(
        os.path.join(static_folder, x)), reverse=True)
    
    return render_template('display.html', images=images, 
                          folder='college_non_mess',
                          title='College Non-Mess Members')

@app.route('/outsiders')
def display_outsiders():
    sync_images(SOURCE_FOLDERS['outsiders'], 'outsiders')
    
    static_folder = 'static/outsiders'
    images = [f for f in os.listdir(static_folder) 
              if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    images.sort(key=lambda x: os.path.getmtime(
        os.path.join(static_folder, x)), reverse=True)
    
    return render_template('display.html', images=images, 
                          folder='outsiders',
                          title='Outsiders')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
