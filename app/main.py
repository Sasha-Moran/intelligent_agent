from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from find_on_wiki import get_information
from predict import predict_user_image


app = Flask(__name__)
app.config["SECRET_KEY"] = "dbsjhsbskds"
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)


class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, "Only images are allowed"),
            FileRequired("File field should not be empty")
        ]
    )
    submit = SubmitField("Upload")


@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"], filename)


@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for("get_file", filename=filename)
        name = predict_user_image(f"uploads/{filename}")
        text, wiki_url = get_information(name)
    else:
        file_url = None
        text = None
        wiki_url = None
        name = None
    return render_template("index.html",
                           form=form,
                           file_url=file_url,
                           context=text,
                           wiki_url=wiki_url,
                           name=name)


if __name__ == "__main__":
    app.run(debug=True, host="192.168.1.102")
