from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import base64

app = Flask(__name__)

# Load the model once
new_model = load_model('models/brainTumorImageClassifier.h5')


# Function to run the model on the image
def model(image):
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return "Unable to process image!"
    
    # Resize image to match model input
    resizedImg = tf.image.resize(img, (256, 256))

    # Predict using the model
    results = new_model.predict(np.expand_dims(resizedImg / 255, 0))
    print(results)
    if results[0][0] > 0.5 : return "Glioma"
    if results[0][1] > 0.5 : return "Meningioma"
    if results[0][2] > 0.5 : return "No Tumor"
    if results[0][3] > 0.5 : return "Pituitary"


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        imageFile = request.files.get('imagefile')
        if imageFile and imageFile.filename:
            imageData = imageFile.read()
            prediction = model(imageData)
            #Need to convert the image to base64 to show on website
            newImg = base64.b64encode(imageData).decode('utf-8')  
            return redirect(url_for("result", results=prediction, image=newImg))

        return redirect(url_for("result", results="not found!"))
    else:
        return render_template("index.html")


@app.route("/results")
def result():
    results = request.args.get("results", "No result provided")
    imageData = request.args.get("image", "None found.")
    explanations = {
        "Glioma": "Gliomas are tumors that are often cancerous but not always. They cover around 33 percent of all brain tumors and stem from glial cells.",
        "Meningioma": "Meningiomas are tumors that are often not cancerous roughly 85-90 percent not. They cover around 30 percent of brain tumors and come from meninges.",
        "No Tumor": "Congrats, no tumor was detected! Your brain is healthy and normal. ",
        "Pituitary": "Pituatary tumors are often mostly not concerous with only a small percent being cancerous. They are caused by abnormal growth of the pituitary gland in the brain and cover roughly 17 percent of brain tumors."
    }

    explanation = explanations.get(results, "No explanation available.")
    
    nextSteps = {
        "Glioma": "The next steps for a glioma tumor depends on the tumor grade, if aggresive like glioblastoma average survival time ranges around 12-18 months. Treatments include surgery, radiation therapy and chemotherapy. Check in with a doctor for accurate information. ",
        "Meningioma": "The next steps for a meningioma tumor depends on many factors, with 5 year survival rates being 95.7 percent for grade I, 81.8 percent for grade II, and 46.7 percent for grade III. Check in with a doctor for accurate information.",
        "No Tumor": "There are no expected complications or health concerns. ",
        "Pituitary": "The next steps for a pituitary tumor depends on many factors, with 5 year survival rates being 97 percent, and 81 percent for cancerous pituitary tumors. Most pituitary tumors are curable through treatment like surgery, but check in with a doctor for accurate information.",
    }

    nextStep = nextSteps.get(results, "No next steps available.")

    return render_template("results.html", content=results, explanation=explanation, image=imageData, nextSteps = nextStep)


if __name__ == "__main__":
    app.run()  # Use debug mode for better error tracking
