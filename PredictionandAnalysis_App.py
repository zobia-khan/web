from flask import Flask,render_template,url_for,request,redirect
from werkzeug.utils import secure_filename #for taking filename only not its path
from datetime import datetime
import os
from PredictionandAnalysisEnglish import pre_process,visualizations,feature_extraction_english,report

UPLOAD_FOLDER ="/Users/Zobia/FYP Web/media/uploaded"
#Allowed file extension is .csv
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS
    

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            #filename = secure_filename(file.filename) #remove the slashes in filename for secure uploading
            #new_filename  = f'{filename.split(".")[0]}_{str(datetime.now())}.csv'
            # # new_filename = new_filename.replace("\\","\")  
            # # file.save(os.path.join('media','/uploaded',new_filename))
            # pre_process(file)
            # process(file)
            # feature_extraction_english(file)
            # report(file)
                        # Get the selected option
            option = request.form.get('option')

            # Call the appropriate function based on the selected option
            if option == 'english':
                pre_process(file)
            elif option == 'urdu':
                handle_urdu(file)        
            elif option == 'roman_urdu':
                handle_roman_urdu(file)
            elif option == 'roman_urdu_mix':
                handle_roman_urdu_mix(file)

            return 'File uploaded and option processed successfully.'
        
    return render_template('upload.html')

# Function to handle Urdu option
def handle_urdu(file):
    # Perform Urdu option logic here
    # Example: Process the uploaded file for Urdu language
    pass

# Function to handle Roman Urdu option
def handle_roman_urdu(file):
    # Perform Roman Urdu option logic here
    # Example: Process the uploaded file for Roman Urdu language
    pass

# Function to handle Roman Urdu Mix option
def handle_roman_urdu_mix(file):
    # Perform Roman Urdu Mix option logic here
    # Example: Process the uploaded file for Roman Urdu Mix language
    pass

if __name__ == "__main__":
    app.run(debug=True)