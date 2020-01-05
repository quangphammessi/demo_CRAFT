import os
import sys
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename

sys.path.append('src')

from test_new import test


print(sys.path)

# ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', image_name='nothing')


# @app.route('/upload-image', methods=['GET', 'POST'])
# def upload_image():
#     return render_template('public/upload_image.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'src/uploads', secure_filename(f.filename))
        print(f.filename)
        f.save(file_path)

        # test(file_path)

        # return 'Done'
    # return render_template('index.html', image_name='0.jpg')
    return jsonify(image_name='0.jpg')


if __name__ == '__main__':
    app.run(debug=True)