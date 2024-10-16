from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from omr import OMR
import traceback  
import json

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Server is reachable"}), 200


@app.route('/hello', methods=['GET'])
def hello_world():
    return "Hello, World!", 200
@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Ensure the image is in the request
        if 'image' not in request.files:
            print("No image found in the request")
            return jsonify({"error": "OMR API: No image in request"}), 400
        
        # Save the image
        file = request.files['image']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        print(f"Image saved to: {file_path}")

        # Ensure answers are in the request
        if 'answers' not in request.form:
            print("No answers found in the request")
            return jsonify({"error": "OMR API: Missing answers in request"}), 400

        # Parse the answers
        answers = request.form['answers']
        print("Received answers as JSON string:", answers)
        answers = json.loads(answers)  # Convert answers from JSON string
        print("Parsed answers:", answers)

        # Validate answers
        if not isinstance(answers, list) or not all(isinstance(ans, int) for ans in answers):
            print("Answers must be a list of integers")
            return jsonify({"error": "OMR API: Answers must be a list of integers"}), 400

        # Read and decode the image
        img = cv2.imread(file_path)
        if img is None:
            print("Failed to read image from file path")
            return jsonify({"error": "Image decoding failed"}), 500

        print("Image successfully decoded from saved file")

        # Call the OMR function
        results = OMR(img, answers, is_data=True)
        
        # Handle invalid image quality case
        if results == "Poor Image Quality":
            return jsonify({"error": "Poor Image Quality. Make sure no shadows are in the answer sheet. Ligthning is important"}), 400
        
        # Handle case where results are None (if the OMR function failed)
        if results is None:
            print("OMR processing failed: Not enough contours detected for the answer sheet.")
            return jsonify({"error": "OMR API: Not enough contours detected for the answer sheet. Make sure to capture the boxes of the answer sheet properly. Do not fold the sheet"}), 400

        # Unpack the results safely
        try:
            set_val, digit_text, score, rating, shaded_answers = results
        except ValueError as e:
            print("Error unpacking OMR results:", str(e))
            return jsonify({"error": "OMR API: Not enough contours detected for the answer sheet. Make sure to capture the boxes of the answer sheet properly. Do not fold the sheet"}), 400


        # Prepare the response
        shaded_answers = [[int(ans) for ans in row] for row in shaded_answers]
        response = {
            "set_val": set_val,
            "digit_text": digit_text,
            "score": int(score),  
            "rating": rating,
            "shaded_answers": shaded_answers,
            "is_data": True
        }

        print("OMR Results:", response)
        return jsonify(response), 200

    except Exception as e:
        print("Exception occurred:", str(e))
        print(traceback.format_exc())  
        return jsonify({"error": str(e)}), 500


# Bind at 192.168.1.99
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)