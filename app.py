from dotenv import load_dotenv
from flask import Flask, request, render_template, redirect, session, url_for, flash
from werkzeug.utils import secure_filename
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage
from io import BytesIO
from huggingface_hub import login
import os
import sqlite3
import pandas as pd

load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['HUGGINGFACE_TOKEN'] = os.getenv('HUGGINGFACE_TOKEN')
app.config['HUGGINGFACE_REPO'] = os.getenv('HUGGINGFACE_REPO')

USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')

print(f'Username : {USERNAME}')
print(f'Password : {PASSWORD}')
print(f'Huggingface Token : {app.config['HUGGINGFACE_TOKEN']}')
print(f'Huggingface Repo : {app.config['HUGGINGFACE_REPO']}')

if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

login(app.config['HUGGINGFACE_TOKEN'])

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instruction TEXT,
            text TEXT,
            image TEXT,
            output TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    if 'username' in session:
        conn = get_db_connection()
        data = conn.execute('SELECT * FROM data').fetchall()
        conn.close()
        return render_template('index.html', data=data)
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        input_user = request.form['username']
        input_pass = request.form['password']

        print(USERNAME, PASSWORD)
        print(input_user, input_pass)

        if input_user == USERNAME and input_pass == PASSWORD:
            session['username'] = input_user
            return redirect(url_for('index'))

        flash("Invalid username or password", "error")
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/store', methods=['POST'])
def store():
    instruction = request.form.get('instruction')
    text = request.form.get('text')
    output = request.form.get('output')
    image_file = request.files.get('image')
    image_path = ''

    if image_file and image_file.filename != '':
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace('\\', '/')
        image_file.save(image_path)

    conn = get_db_connection()
    conn.execute('INSERT INTO data (instruction, text, image, output) VALUES (?, ?, ?, ?)',
                 (instruction, text, image_path, output))
    conn.commit()
    conn.close()

    flash("Successfully stored!", "success")
    return redirect(url_for('index'))

@app.route('/update/<int:id>', methods=['POST'])
def update(id):
    instruction = request.form.get('instruction')
    text = request.form.get('text')
    output = request.form.get('output')
    image_file = request.files.get('image')

    conn = get_db_connection()
    old_data = conn.execute('SELECT image FROM data WHERE id = ?', (id,)).fetchone()

    if image_file and image_file.filename != '':
        if old_data and os.path.exists(old_data['image']):
            os.remove(old_data['image'])
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)
        conn.execute('UPDATE data SET instruction = ?, text = ?, image = ?, output = ? WHERE id = ?',
                     (instruction, text, image_path, output, id))
    else:
        conn.execute('UPDATE data SET instruction = ?, text = ?, output = ? WHERE id = ?',
                     (instruction, text, output, id))

    conn.commit()
    conn.close()

    flash("Data successfully updated!", "success")
    return redirect(url_for('index'))

@app.route('/delete/<int:id>', methods=['POST'])
def delete(id):
    conn = get_db_connection()
    data = conn.execute('SELECT image FROM data WHERE id = ?', (id,)).fetchone()

    if data and os.path.exists(data['image']):
        os.remove(data['image'])

    conn.execute('DELETE FROM data WHERE id = ?', (id,))
    conn.commit()
    conn.close()

    flash("Data successfully deleted!", "success")
    return redirect(url_for('index'))

@app.route('/parquet', methods=['GET'])
def parquet():
    conn = get_db_connection()
    rows = conn.execute('SELECT instruction, text, image, output FROM data').fetchall()
    conn.close()

    instructions, texts, outputs, images = [], [], [], []

    for row in rows:
        instructions.append(row['instruction'] or "")
        texts.append(row['text'] or "")
        outputs.append(row['output'] or "")

        if row['image'] and os.path.exists(row['image']):
            with PILImage.open(row['image']) as image:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                buffer = BytesIO()
                image.save(buffer, format='JPEG')
                images.append(buffer.getvalue())
        else:
            images.append(None)

    features = Features({
        'instruction': Value('string'),
        'text': Value('string'),
        'image': Image(),
        'output': Value('string'),
    })

    df = pd.DataFrame({
        'instruction': instructions,
        'text': texts,
        'image': images,
        'output': outputs
    })

    try:
        dataset = Dataset.from_pandas(df, features=features)
        dataset.push_to_hub(app.config['HUGGINGFACE_REPO'])
        flash("Successfully stored to HuggingFace!", "success")
    except Exception as e:
        print(e)
        flash("Failed to store to HuggingFace!", "error")

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)