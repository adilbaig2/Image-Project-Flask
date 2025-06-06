import numpy as np
import json
from flask import Flask, render_template, request, send_file, redirect, url_for, flash, session
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
import os
from sklearn.metrics import roc_curve
import scipy.stats
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import base64
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_pro = db.Column(db.Boolean, default=False)
    subscription = db.relationship('Subscription', backref='user', uselist=False)
    gallery_images = db.relationship('GalleryImage', backref='user', lazy=True)
    sample_images = db.relationship('SampleImage', backref='user', lazy=True)

class Subscription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    payment_method = db.Column(db.String(20), nullable=False)  # 'card' or 'cash'
    amount = db.Column(db.Float, nullable=False)
    payment_date = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())
    card_details = db.Column(db.String(500))  # Encrypted card details for card payments
    billing_address = db.Column(db.String(500))  # Encrypted billing address for card payments

class Embedding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model_type = db.Column(db.String(20), nullable=False)  # 'clip' or 'autoencoder'
    created_at = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())
    binary_file = db.Column(db.String(200), nullable=False)
    original_filename = db.Column(db.String(200), nullable=False)

class GalleryImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class SampleImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Pro mode required decorator
def pro_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or not user.is_pro:
            flash('This feature requires a Pro subscription.', 'error')
            return redirect(url_for('subscription'))
        return f(*args, **kwargs)
    return decorated_function

# Create database tables
def init_db():
    with app.app_context():
        # Drop all existing tables
        db.drop_all()
        # Create all tables
        db.create_all()
        print("Database initialized successfully!")

# Initialize the database
init_db()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embedding = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
    embedding = embedding.numpy()[0]
    return embedding

def compare_images(image1, image2, threshold):
    embedding1 = compute_embedding(image1)
    embedding2 = compute_embedding(image2)
    sim_score = cosine_similarity([embedding1], [embedding2])[0][0] * 100
    is_similar = sim_score > threshold
    return f"{sim_score:.2f}%", "Similar" if is_similar else "Not Similar"

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    similarity = result = None

    if request.method == 'POST':
        file1 = request.files['image1']
        file2 = request.files['image2']
        threshold = float(request.form['threshold'])

        if file1 and file2:
            image1 = Image.open(io.BytesIO(file1.read()))
            image2 = Image.open(io.BytesIO(file2.read()))
            similarity, result = compare_images(image1, image2, threshold)

    return render_template('index.html', similarity=similarity, result=result)

@app.route('/upload_gallery', methods=['POST'])
@login_required
def upload_gallery():
    if 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            user_id = session['user_id']
            image_dir = f'static/gallery/user_{user_id}'
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)

            filename = file.filename
            file.save(os.path.join(image_dir, filename))
            
            # Save to database
            new_image = GalleryImage(filename=filename, user_id=user_id)
            db.session.add(new_image)
            db.session.commit()
            
    return redirect(url_for('gallery'))

@app.route('/gallery', methods=['GET', 'POST'])
@login_required
def gallery():
    user_id = session['user_id']
    image_dir = f'static/gallery/user_{user_id}'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Get images from database
    user_images = GalleryImage.query.filter_by(user_id=user_id).all()
    images = [img.filename for img in user_images if os.path.exists(os.path.join(image_dir, img.filename))]
    images.sort(key=lambda x: os.path.getmtime(os.path.join(image_dir, x)), reverse=True)
    
    return render_template('gallery.html', images=images, user_id=user_id)

@app.route('/delete_last', methods=['POST'])
@login_required
def delete_last():
    user_id = session['user_id']
    image_dir = f'static/gallery/user_{user_id}'
    
    # Get last image from database
    last_image = GalleryImage.query.filter_by(user_id=user_id).order_by(GalleryImage.id.desc()).first()
    if last_image:
        os.remove(os.path.join(image_dir, last_image.filename))
        db.session.delete(last_image)
        db.session.commit()
        
    return redirect(url_for('gallery'))

@app.route('/delete_by_name', methods=['POST'])
@login_required
def delete_by_name():
    user_id = session['user_id']
    filename = request.form.get('filename')
    image_path = os.path.join(f'static/gallery/user_{user_id}', filename)
    
    if os.path.exists(image_path):
        os.remove(image_path)
        # Delete from database
        image = GalleryImage.query.filter_by(user_id=user_id, filename=filename).first()
        if image:
            db.session.delete(image)
            db.session.commit()
            
    return redirect(url_for('gallery'))

@app.route('/upload_sample', methods=['POST'])
@login_required
def upload_sample():
    message = None
    sample_image = None
    user_id = session['user_id']
    model_type = request.form.get('model_type', 'clip')
    session['last_sample_model_type'] = model_type
    
    # Check if user is trying to use autoencoder without pro
    if model_type == 'autoencoder' and not session.get('is_pro'):
        flash('Autoencoder model is only available for Pro users.', 'error')
        return redirect(url_for('subscription'))
    
    if 'sample_image' in request.files:
        file = request.files['sample_image']
        if file.filename != '':
            # Validate file type
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
            if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, or GIF).', 'error')
                return redirect(url_for('sample'))
            
            sample_dir = f'static/sample/user_{user_id}'
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
                
            # Clear existing sample images
            existing_images = SampleImage.query.filter_by(user_id=user_id).all()
            for img in existing_images:
                try:
                    os.remove(os.path.join(sample_dir, img.filename))
                except:
                    pass  # Ignore if file doesn't exist
                db.session.delete(img)
            db.session.commit()
            
            # Save new sample image with timestamp to ensure uniqueness
            timestamp = int(time.time())
            filename = f"sample_{timestamp}{os.path.splitext(file.filename)[1]}"
            file_path = os.path.join(sample_dir, filename)
            file.save(file_path)
            
            # Save to database
            new_sample = SampleImage(filename=filename, user_id=user_id)
            db.session.add(new_sample)
            db.session.commit()
            
            message = f"Sample image uploaded successfully! Using {model_type.upper()} model for comparison."
            sample_image = filename
            flash(message, 'success')
        else:
            flash('No file selected.', 'error')
    else:
        flash('No file uploaded.', 'error')
        
    return redirect(url_for('sample'))

@app.route('/delete_sample', methods=['POST'])
@login_required
def delete_sample():
    user_id = session['user_id']
    sample_dir = f'static/sample/user_{user_id}'
    
    # Get current sample image
    sample_record = SampleImage.query.filter_by(user_id=user_id).first()
    if sample_record:
        try:
            # Remove file from filesystem
            os.remove(os.path.join(sample_dir, sample_record.filename))
        except:
            pass  # Ignore if file doesn't exist
            
        # Remove from database
        db.session.delete(sample_record)
        db.session.commit()
        flash('Sample image deleted successfully.', 'success')
    else:
        flash('No sample image found.', 'error')
        
    return redirect(url_for('sample'))

@app.route('/sample', methods=['GET', 'POST'])
@login_required
def sample():
    user_id = session['user_id']
    sample_image = None
    sample_dir = f'static/sample/user_{user_id}'
    selected_model = session.get('last_sample_model_type', 'clip')
    
    if request.method == 'POST':
        model_type = request.form.get('model_type', 'clip')
        session['last_sample_model_type'] = model_type
        selected_model = model_type
        
        # Check if user is trying to use autoencoder without pro
        if model_type == 'autoencoder' and not session.get('is_pro'):
            flash('Autoencoder model is only available for Pro users.', 'error')
            return redirect(url_for('subscription'))
    
    if os.path.exists(sample_dir):
        sample_record = SampleImage.query.filter_by(user_id=user_id).first()
        if sample_record:
            sample_image = sample_record.filename
            
    return render_template('sample.html', 
                         sample_image=sample_image, 
                         user_id=user_id, 
                         selected_model=selected_model)

@app.route('/subscription', methods=['GET', 'POST'])
@login_required
def subscription():
    user = User.query.get(session['user_id'])
    if not user:
        flash('User not found. Please log in again.', 'error')
        session.clear()
        return redirect(url_for('login'))
        
    if user.is_pro:
        flash('You already have a Pro subscription!', 'info')
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        payment_method = request.form.get('payment_method')
        amount = float(request.form.get('amount', 0))
        
        if payment_method == 'card':
            # In a real application, you would validate and process the card details
            # For this example, we'll just store them (in production, use proper encryption)
            card_number = request.form.get('card_number')
            expiry = request.form.get('expiry')
            cvv = request.form.get('cvv')
            billing_address = request.form.get('billing_address')
            
            # Mock card validation
            if not all([card_number, expiry, cvv, billing_address]):
                flash('Please fill in all card details.', 'error')
                return redirect(url_for('subscription'))
                
            # Create subscription record
            subscription = Subscription(
                user_id=user.id,
                payment_method='card',
                amount=amount,
                card_details=f"{card_number[-4:]}-{expiry}-{cvv}",  # Store only last 4 digits in production
                billing_address=billing_address
            )
            
        elif payment_method == 'cash':
            if amount <= 0:
                flash('Please enter a valid amount.', 'error')
                return redirect(url_for('subscription'))
                
            subscription = Subscription(
                user_id=user.id,
                payment_method='cash',
                amount=amount
            )
            
        else:
            flash('Invalid payment method.', 'error')
            return redirect(url_for('subscription'))
            
        try:
            db.session.add(subscription)
            user.is_pro = True
            db.session.commit()
            session['is_pro'] = True  # Update session to reflect pro status
            flash('Thank you for subscribing! You now have access to Pro features.', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred while processing your payment. Please try again.', 'error')
            
    return render_template('subscription.html')

@app.route('/pro_features')
@pro_required
def pro_features():
    user = User.query.get(session['user_id'])
    return render_template('pro_features.html', user=user)

@app.route('/binary', methods=['GET', 'POST'])
@login_required
def binary():
    user_id = session['user_id']
    user_binary_dir = f'static/binary/user_{user_id}'
    if not os.path.exists(user_binary_dir):
        os.makedirs(user_binary_dir)

    # List all binary files for the user with their model types
    binary_files = []
    for f in os.listdir(user_binary_dir):
        if f.endswith('.txt'):
            embedding = Embedding.query.filter_by(
                user_id=user_id,
                binary_file=f
            ).first()
            model_type = None
            original_filename = f
            if embedding:
                model_type = embedding.model_type
                original_filename = embedding.original_filename
            else:
                # Try to read model_type and original_filename from file
                try:
                    with open(os.path.join(user_binary_dir, f), 'r') as file_check:
                        data = json.load(file_check)
                        model_type = data.get('model_type')
                        if data.get('original_filename'):
                            original_filename = data.get('original_filename')
                        if not model_type:
                            lower_name = f.lower()
                            if 'clip' in lower_name:
                                model_type = 'clip'
                            else:
                                model_type = 'autoencoder'
                except Exception:
                    lower_name = f.lower()
                    if 'clip' in lower_name:
                        model_type = 'clip'
                    else:
                        model_type = 'autoencoder'
            binary_files.append({
                'filename': f,
                'model_type': model_type,
                'original_filename': original_filename
            })
    
    # Get model type from form if provided, default to CLIP for free users
    model_type = request.form.get('model_type', 'clip')
    if not session.get('is_pro'):
        model_type = 'clip'  # Force CLIP for free users
    
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                try:
                    # Read and process the image
                    image = Image.open(io.BytesIO(file.read()))
                    # Use original filename for binary file (no model type in name)
                    original_filename = file.filename
                    base_name = os.path.splitext(original_filename)[0]
                    binary_filename = f"{base_name}.txt"
                    binary_path = os.path.join(user_binary_dir, binary_filename)
                    # Convert image to binary and save (pass model_type and original_filename)
                    convert_to_color_binary(image, binary_path, model_type=model_type, original_filename=original_filename)
                    # Save embedding record
                    embedding = Embedding(
                        user_id=user_id,
                        model_type=model_type,
                        binary_file=binary_filename,
                        original_filename=original_filename
                    )
                    db.session.add(embedding)
                    db.session.commit()
                    flash('Image successfully converted to binary!', 'success')
                    return redirect(url_for('binary'))
                except Exception as e:
                    flash(f'Error converting image: {str(e)}', 'error')
                    return redirect(url_for('binary'))

    return render_template('binary.html', binary_files=binary_files, user_id=user_id, model_type=model_type)

def convert_to_color_binary(image, out_file=None, model_type='clip', original_filename=None):
    try:
        # Convert image to RGB
        image = image.convert("RGB")
        np_img = np.array(image)
        height, width, channels = np_img.shape
        flat = np_img.flatten()
        binary_strings = [format(val, '08b') for val in flat]
        binary_data = ''.join(binary_strings)

        payload = {
            "shape": [height, width, channels],
            "binary": binary_data,
            "original_filename": original_filename,
            "model_type": model_type
        }

        if out_file is None:
            out_file = "color_binary_output.txt"
            
        with open(out_file, "w") as f:
            json.dump(payload, f)

        return out_file
    except Exception as e:
        print(f"Error in convert_to_color_binary: {str(e)}")
        raise

def reverse_color_binary(file_path, output_name=None, user_id=None):
    with open(file_path, "r") as f:
        data = json.load(f)
    shape = tuple(data["shape"])
    binary_str = data["binary"]
    original_filename = data.get("original_filename")
    if not original_filename:
        original_filename = os.path.splitext(os.path.basename(file_path))[0] + ".png"
    base_name = os.path.splitext(original_filename)[0]
    total_pixels = shape[0] * shape[1] * shape[2]
    if len(binary_str) != total_pixels * 8:
        raise ValueError("Binary string length does not match expected image size.")
    bytes_array = [int(binary_str[i:i+8], 2) for i in range(0, len(binary_str), 8)]
    img_array = np.array(bytes_array, dtype=np.uint8).reshape(shape)
    img = Image.fromarray(img_array, mode="RGB")
    user_dir = f'static/reconstructed/user_{user_id}' if user_id else 'static/reconstructed'
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    # Use provided output_name or generate one from original filename (no model type in name)
    if output_name is None:
        output_name = f"{base_name}.png"
    output_path = os.path.join(user_dir, output_name)
    img.save(output_path)
    # Return the path relative to static/
    return f"reconstructed/user_{user_id}/{output_name}" if user_id else output_path

@app.route('/delete_binary', methods=['POST'])
@login_required
def delete_binary():
    user_id = session['user_id']
    user_binary_dir = f'static/binary/user_{user_id}'
    filename = request.form.get('filename')
    file_path = os.path.join(user_binary_dir, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect(url_for('binary'))

@app.route('/reverse', methods=['GET', 'POST'])
@login_required
def reverse():
    user_id = session['user_id']
    user_binary_dir = f'static/binary/user_{user_id}'
    user_reverse_dir = f'static/reconstructed/user_{user_id}'
    if not os.path.exists(user_reverse_dir):
        os.makedirs(user_reverse_dir)
    if not os.path.exists(user_binary_dir):
        os.makedirs(user_binary_dir)

    # Get binary files with their model types
    binary_files = []
    for f in os.listdir(user_binary_dir):
        if f.endswith('.txt'):
            embedding = Embedding.query.filter_by(
                user_id=user_id,
                binary_file=f
            ).first()
            model_type = None
            original_filename = f
            if embedding:
                model_type = embedding.model_type
                original_filename = embedding.original_filename
            else:
                lower_name = f.lower()
                if 'clip' in lower_name:
                    model_type = 'clip'
                else:
                    model_type = 'autoencoder'
                original_filename = f
            binary_files.append({
                'filename': f,
                'model_type': model_type,
                'original_filename': original_filename
            })

    selected_file = None
    image_path = None
    error = None
    process_mode = None
    reconstructed_filename = None
    selected_model_type = None
    clip_embedding = None
    clip_comparison = None

    if request.method == 'POST':
        if 'binaryfile' in request.files:
            file = request.files.get('binaryfile')
            if file and file.filename.endswith('.txt'):
                upload_path = os.path.join(user_reverse_dir, file.filename)
                file.save(upload_path)
                try:
                    # Always read model_type from the uploaded text file
                    with open(upload_path, 'r') as file_check:
                        data = json.load(file_check)
                        model_type = data.get('model_type')
                        if not model_type:
                            lower_name = file.filename.lower()
                            if 'clip' in lower_name:
                                model_type = 'clip'
                            else:
                                model_type = 'autoencoder'
                        original_filename = data.get('original_filename')
                        if not original_filename:
                            original_filename = os.path.splitext(os.path.basename(upload_path))[0]
                    selected_model_type = model_type
                    base_name = os.path.splitext(original_filename)[0]
                    reconstructed_filename = f"{base_name}.png"
                    image_path = reverse_color_binary(upload_path, output_name=reconstructed_filename, user_id=user_id)
                    process_mode = 'center'
                except Exception as e:
                    error = f"Error processing file: {e}"
            else:
                error = "Invalid file type. Please upload a .txt file."
        elif 'selected_file' in request.form:
            selected_file = request.form.get('selected_file')
            if selected_file:
                file_path = os.path.join(user_binary_dir, selected_file)
                try:
                    # Always read model_type from the text file
                    with open(file_path, 'r') as file_check:
                        data = json.load(file_check)
                        model_type = data.get('model_type')
                        if not model_type:
                            lower_name = selected_file.lower()
                            if 'clip' in lower_name:
                                model_type = 'clip'
                            else:
                                model_type = 'autoencoder'
                        original_filename = data.get('original_filename')
                        if not original_filename:
                            original_filename = os.path.splitext(os.path.basename(file_path))[0]
                    selected_model_type = model_type
                    base_name = os.path.splitext(original_filename)[0]
                    reconstructed_filename = f"{base_name}.png"
                    image_path = reverse_color_binary(file_path, output_name=reconstructed_filename, user_id=user_id)
                    process_mode = 'left'
                except Exception as e:
                    error = f"Error processing file: {e}"

    return render_template('reverse.html', 
                         binary_files=binary_files, 
                         selected_file=selected_file, 
                         image_path=image_path, 
                         error=error, 
                         process_mode=process_mode,
                         reconstructed_filename=reconstructed_filename,
                         selected_model_type=selected_model_type,
                         clip_embedding=clip_embedding,
                         clip_comparison=clip_comparison)

@app.route('/download_binary/<filename>')
@login_required
def download_binary(filename):
    user_id = session['user_id']
    user_binary_dir = f'static/binary/user_{user_id}'
    file_path = os.path.join(user_binary_dir, filename)
    
    # Get custom filename from query parameter or use original
    custom_filename = request.args.get('custom_filename')
    if not custom_filename:
        # Try to get original filename from database
        embedding = Embedding.query.filter_by(
            user_id=user_id,
            binary_file=filename
        ).first()
        if embedding:
            custom_filename = embedding.original_filename
        else:
            custom_filename = filename
    
    # Ensure the filename has .txt extension
    if not custom_filename.endswith('.txt'):
        custom_filename += '.txt'
    
    return send_file(file_path, as_attachment=True, download_name=custom_filename)

@app.route('/run_image_check', methods=['POST'])
@login_required
def run_image_check():
    user_id = session['user_id']
    sample_dir = f'static/sample/user_{user_id}'
    gallery_dir = f'static/gallery/user_{user_id}'
    binary_dir = f'static/binary/user_{user_id}'
    
    if not os.path.exists(binary_dir):
        os.makedirs(binary_dir)
        
    check_results = []
    binary_files = []
    
    # Get selected model type from form
    model_type = request.form.get('model_type', 'clip')
    session['last_sample_model_type'] = model_type
    
    # Check if user is trying to use autoencoder without pro
    if model_type == 'autoencoder' and not session.get('is_pro'):
        flash('Autoencoder model is only available for Pro users.', 'error')
        return redirect(url_for('subscription'))
    
    # Get sample image
    sample_record = SampleImage.query.filter_by(user_id=user_id).first()
    if not sample_record:
        check_results.append("No sample image found to compare.")
        return render_template('sample.html', 
                             check_results=check_results, 
                             user_id=user_id, 
                             selected_model=model_type)
                              
    sample_image_path = os.path.join(sample_dir, sample_record.filename)
    sample_image = Image.open(sample_image_path)
    
    # Get gallery images
    user_images = GalleryImage.query.filter_by(user_id=user_id).all()
    if not user_images:
        check_results.append("No images in gallery to compare with.")
        return render_template('sample.html', 
                             check_results=check_results, 
                             user_id=user_id, 
                             selected_model=model_type)
                              
    converted_count = 0
    for img_record in user_images:
        gallery_image_path = os.path.join(gallery_dir, img_record.filename)
        gallery_image = Image.open(gallery_image_path)
        
        # Compare images using the selected model
        similarity, result = compare_images(sample_image, gallery_image, 80)
        sim_value = float(similarity.replace('%',''))
        result_str = f"{img_record.filename}: {similarity} - {result}"
        check_results.append(result_str)
        
        if sim_value > 80:
            # Generate binary filename based on original image name
            base_name = os.path.splitext(img_record.filename)[0]
            binary_filename = f"{base_name}.txt"
            binary_file_path = os.path.join(binary_dir, binary_filename)
            
            # Convert to binary using the selected model, always pass model_type and original_filename
            if model_type == 'autoencoder':
                convert_to_autoencoder_binary(gallery_image, binary_file_path, model_type='autoencoder', original_filename=img_record.filename)
            else:  # default to CLIP
                convert_to_color_binary(gallery_image, binary_file_path, model_type='clip', original_filename=img_record.filename)
                
            binary_files.append(binary_file_path)
            
            # Save embedding record with model type
            embedding = Embedding(
                user_id=user_id,
                model_type=model_type,
                binary_file=binary_filename,
                original_filename=img_record.filename
            )
            db.session.add(embedding)
            db.session.commit()
            
            # Remove original gallery image and record
            os.remove(gallery_image_path)
            db.session.delete(img_record)
            db.session.commit()
            converted_count += 1
            
    if converted_count > 0:
        check_results.append(f"{converted_count} file(s) were converted into binary using the {model_type.upper()} model.")
    else:
        check_results.append("No files were converted into binary.")
        
    return render_template('sample.html', 
                         check_results=check_results, 
                         user_id=user_id, 
                         selected_model=model_type)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different username.', 'error')
            return redirect(url_for('signup'))
            
        if User.query.filter_by(email=email).first():
            flash('Email already exists. Please use a different email.', 'error')
            return redirect(url_for('signup'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred. Please try again.', 'error')
            
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_pro'] = user.is_pro
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password.', 'error')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('is_pro', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

def get_embedding_info(filename):
    """Helper function to get embedding information for a given file."""
    try:
        embedding = Embedding.query.filter_by(binary_file=filename).first()
        return embedding
    except Exception as e:
        print(f"Error getting embedding info: {str(e)}")
        return None

# Add the helper function to the template context
@app.context_processor
def utility_processor():
    return dict(get_embedding_info=get_embedding_info)

@app.route('/download_reconstructed/<filename>')
@login_required
def download_reconstructed(filename):
    user_id = session['user_id']
    user_reverse_dir = f'static/reconstructed/user_{user_id}'
    file_path = os.path.join(user_reverse_dir, filename)
    
    # Get custom filename from query parameter or use original
    custom_filename = request.args.get('custom_filename')
    if not custom_filename:
        # Try to get original filename from database
        base_name = os.path.splitext(filename)[0]
        embedding = Embedding.query.filter_by(
            user_id=user_id,
            binary_file=f"{base_name}_binary.txt"
        ).first()
        if embedding:
            custom_filename = os.path.splitext(embedding.original_filename)[0] + '.png'
        else:
            custom_filename = filename
    
    # Ensure the filename has .png extension
    if not custom_filename.endswith('.png'):
        custom_filename += '.png'
    
    return send_file(file_path, as_attachment=True, download_name=custom_filename)

def reverse_autoencoder_binary(file_path, output_name=None, user_id=None):
    try:
        # Use the same logic as reverse_color_binary, but set model_type to 'autoencoder'
        return reverse_color_binary(file_path, output_name, user_id)
    except Exception as e:
        print(f"Error in reverse_autoencoder_binary: {str(e)}")
        return None, f"Error reconstructing autoencoder image: {str(e)}"

def reverse_vits_binary(file_path, output_name=None, user_id=None):
    try:
        # Use the same logic as reverse_color_binary, but set model_type to 'vits'
        return reverse_color_binary(file_path, output_name, user_id)
    except Exception as e:
        print(f"Error in reverse_vits_binary: {str(e)}")
        return None, f"Error reconstructing VITS image: {str(e)}"

def convert_to_autoencoder_binary(image, out_file=None, model_type='autoencoder', original_filename=None):
    # Use the same as convert_to_color_binary, but pass model_type and original_filename
    return convert_to_color_binary(image, out_file, model_type=model_type, original_filename=original_filename)

if __name__ == '__main__':
    app.run(debug=True)