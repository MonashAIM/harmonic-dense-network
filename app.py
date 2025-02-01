import torch
import gzip
import io
from flask import Flask, request, render_template, send_file, session
from flask_session import Session
import torchvision.transforms as transforms
from monai.transforms import CropForeground, Spacing, NormalizeIntensity, Resize
from src.models.FCHardnet import FCHardnet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
import nibabel as nib
import time
from PIL import Image
import torch
from pathlib import Path

app = Flask(__name__, static_url_path="", static_folder="/")
# Check Configuration section for more details
SESSION_TYPE = "filesystem"
app.config.from_object(__name__)
Session(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pre_trans = Compose(
    [
        CropForeground(),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        Spacing(pixdim=(1.2, 1.2), mode=("nearest", "nearest")
        ),
        transforms.Resize(spatial_size=(128, 128)),
    ]
)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

model = FCHardnet(n_classes=1, in_channels=1).to(device)
model.load_state_dict(torch.load("./weights/best_metric.pth"))

@app.route("/", methods=["GET"])
def main():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f_stream = request.files["file"].stream
    img = Image.open(fp=f_stream).convert("L")
    filename = request.files["file"].filename
    upload_image_path = f"./input/{filename}"
    session["upload_image_path"] = upload_image_path
    img.save(upload_image_path, format="PNG")
    return {"upload_image_path": upload_image_path}


@app.route("/predict/2d", methods=["GET"])
def predict_2d_endpoint():
    img = Image.open(session["upload_image_path"])
    tensor = (transforms.PILToTensor()(img)).to(device).unsqueeze_(0).float()
    output = model(tensor)
    post_out = post_trans(output).detach().cpu()
    image = transforms.ToPILImage()(post_out.squeeze_())
    image.save("./output/result.jpeg", format="JPEG")
    return {"predict_path": "./output/result.jpeg"}


@app.route("/download/predict/2d", methods=["GET"])
def download_predict_2d_endpoint():
    return send_file("./output/result.jpeg", as_attachment=True, mimetype="image/jpeg")


@app.route("/predict/3d", methods=["POST"])
def predict_3d_endpoint():
    start = time.time()
    f = request.files["file"].read()
    compressed_stream = io.BytesIO(f.stream)
    with gzip.GzipFile(fileobj=compressed_stream, mode="rb") as decompressed_stream:
        nifti_data = io.BytesIO(decompressed_stream)
    f.close()
    nifti_image = nib.Nifti1Image.from_bytes(nifti_data.read())

    # Saving the nifti image
    print(request.files["file"].filename)

    return {"message": "OK", "upload in (s)": time.time() - start}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
