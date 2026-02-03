from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from .models import Drawing

# Load trained model
model = load_model("parkinsons_model.keras")

def predict_parkinsons(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128,128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    prediction = model.predict(img)[0][0]
    return "Parkinson's" if prediction > 0.5 else "Healthy"

def upload_image(request):
    if request.method == "POST":
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)

        # Predict Parkinson’s
        prediction = predict_parkinsons(fs.path(file_path))

        # Save record
        drawing = Drawing(image=uploaded_file, prediction=prediction)
        drawing.save()

        return render(request, "result.html", {"drawing": drawing})

    return render(request, "upload.html")
