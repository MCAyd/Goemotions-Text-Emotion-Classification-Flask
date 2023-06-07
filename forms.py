from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from flask_ckeditor import CKEditorField
from wtforms.validators import InputRequired


class InputForm(FlaskForm):
	content = CKEditorField('Type your sentence here', validators=[InputRequired()])
	submit = SubmitField('Submit')