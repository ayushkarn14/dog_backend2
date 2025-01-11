from flask import Flask, request, send_file, render_template_string
from ultralytics import YOLO
import os
from io import BytesIO
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = YOLO("./yolomodel.pt")


@app.route("/")
def index():
    with open("index.html", "r") as file:
        content = file.read()
    return render_template_string(content)


@app.route("/process-image", methods=["POST"])
@app.route("/process-image", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return "No image file found in the request", 400

    # Get the image from the request
    image_file = request.files["image"]
    image = Image.open(image_file)

    # Process the image with YOLO
    results = model(image)
    processed_image = results[0].plot()  # This returns a NumPy array

    # Convert the NumPy array to a PIL Image
    processed_image = Image.fromarray(processed_image)

    # Save the processed image to a BytesIO object
    img_io = BytesIO()
    processed_image.save(img_io, "JPEG")
    img_io.seek(0)

    # Send the image back as a response
    return send_file(img_io, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
