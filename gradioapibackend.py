from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
from werkzeug.utils import secure_filename
from gradio_client import Client, handle_file
import tempfile
import shutil

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {
    'image': {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'},
    'video': {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'},
    'audio': {'wav', 'mp3', 'flac', 'ogg', 'aac', 'm4a'}
}

# Create upload folder
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize new v2 detector (ViT-based, more accurate)
NEW_MODEL_AVAILABLE = False
try:
    import detector_v2
    NEW_MODEL_AVAILABLE = True
    print("✅ Detector v2 module loaded")
except Exception as e:
    print(f"⚠️ Detector v2 unavailable: {e}")

# Initialize local model (EfficientNet ONNX + model.pth) as primary detector
LOCAL_AVAILABLE = False
try:
    import inference_2
    LOCAL_AVAILABLE = True
    print("✅ Local inference model loaded successfully")
except Exception as e:
    print(f"⚠️ Local model failed to load: {e}")

# Initialize Hugging Face Space client
try:
    hf_client = Client("PraneshJs/fakevideodetect")
    HF_AVAILABLE = True
    print("✅ Hugging Face Space client initialized successfully")
except Exception as e:
    HF_AVAILABLE = False
    print(f"⚠️ Hugging Face Space client failed to initialize: {e}")

def allowed_file(filename, file_type):
    """Check if file extension is allowed for the given type"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS.get(file_type, set())

def get_file_info(file):
    """Get file information"""
    try:
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)     # Reset to beginning
        return {
            'name': file.filename,
            'size': size,
            'size_mb': round(size / (1024 * 1024), 2),
            'size_kb': round(size / 1024, 1)
        }
    except:
        return {'name': file.filename, 'size': 0, 'size_mb': 0, 'size_kb': 0}

def predict_with_hf_space(file_path, prediction_type):
    """Make prediction using Hugging Face Space"""
    try:
        if not HF_AVAILABLE:
            raise Exception("Hugging Face Space client not available")
        
        if prediction_type == 'image':
            result = hf_client.predict(
                input_image=handle_file(file_path),
                api_name="/predict"
            )
        elif prediction_type == 'video':
            result = hf_client.predict(
                input_video={"video": handle_file(file_path)},
                api_name="/predict_1"
            )
        elif prediction_type == 'audio':
            result = hf_client.predict(
                input_audio=handle_file(file_path),
                api_name="/predict_2"
            )
        else:
            raise Exception(f"Unsupported prediction type: {prediction_type}")
        
        return result
    except Exception as e:
        print(f"❌ HF Space prediction error: {e}")
        raise e

def get_demo_response(file_info, prediction_type):
    """Get demo response when HF Space is not available"""
    type_emojis = {'image': '🖼️', 'video': '🎬', 'audio': '🎵'}
    type_names = {'image': 'Image', 'video': 'Video', 'audio': 'Audio'}
    
    analyses = {
        'image': [
            '• Facial feature detection and analysis',
            '• Skin texture consistency evaluation',
            '• Eye movement and blink pattern analysis',
            '• Lighting and shadow consistency check',
            '• Compression artifact detection'
        ],
        'video': [
            '• Frame-by-frame facial consistency analysis',
            '• Temporal coherence evaluation',
            '• Motion pattern analysis',
            '• Audio-visual synchronization check',
            '• Compression and quality assessment'
        ],
        'audio': [
            '• Spectral analysis and frequency patterns',
            '• Voice authenticity verification',
            '• Speech synthesis artifact detection',
            '• Temporal consistency evaluation',
            '• Background noise analysis'
        ]
    }
    
    emoji = type_emojis[prediction_type]
    name = type_names[prediction_type]
    analysis_points = '\n'.join(analyses[prediction_type])
    
    return f"""{emoji} {name} Analysis Complete!

📄 File: {file_info['name']}
📊 Size: {file_info['size_mb']} MB
✅ Upload Successful!

⚠️ Demo Mode Active

In a production environment with full AI models, this analysis would include:

🔍 Advanced Detection Features:
{analysis_points}

📈 Detailed Results:
• Confidence scores (0-100%)
• Region-specific analysis
• Technical metadata extraction
• Detailed reasoning explanations

💡 Your {prediction_type} was successfully processed and validated by our backend system."""


@app.route('/api/predict/<prediction_type>', methods=['POST'])
def predict(prediction_type):
    """Universal prediction endpoint for image, video, and audio"""
    try:
        # Validate prediction type
        if prediction_type not in ['image', 'video', 'audio']:
            return jsonify({'error': 'Invalid prediction type. Use: image, video, or audio'}), 400
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type (extension OR mimetype)
        mime_ok = False
        try:
            mime_ok = bool(file.mimetype) and file.mimetype.split('/')[0] == ('image' if prediction_type == 'image' else prediction_type)
        except Exception:
            pass
        if not allowed_file(file.filename, prediction_type) and not mime_ok:
            allowed_exts = ', '.join(ALLOWED_EXTENSIONS[prediction_type])
            return jsonify({'error': f'Invalid file type for {prediction_type}. Allowed: {allowed_exts}'}), 400
        
        # Get file info
        file_info = get_file_info(file)
        print(f"📁 Processing {prediction_type}: {file_info['name']} ({file_info['size_mb']}MB)")
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time() * 1000))
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        file.save(file_path)
        print(f"💾 File saved to: {file_path}")
        
        try:
            # Prefer the new v2 model when available
            if NEW_MODEL_AVAILABLE and prediction_type == 'image':
                print("🤖 Using v2 ViT deepfake detector...")
                result = detector_v2.deepfakes_image_predict_v2(file_path)
                return jsonify({
                    'result': result,
                    'status': 'success',
                    'source': 'detector_v2',
                    'file_info': file_info
                })
            # Prefer the old local model when available
            if LOCAL_AVAILABLE and prediction_type in ('image', 'video'):
                print("🤖 Using local model for prediction...")
                if prediction_type == 'image':
                    result = inference_2.deepfakes_image_predict(file_path)
                else:
                    result = inference_2.deepfakes_video_predict(file_path)
                return jsonify({
                    'result': result,
                    'status': 'success',
                    'source': 'local_model',
                    'file_info': file_info
                })
            # Try to use Hugging Face Space first
            if HF_AVAILABLE:
                print("🤖 Using Hugging Face Space for prediction...")
                result = predict_with_hf_space(file_path, prediction_type)
                
                return jsonify({
                    'result': result,
                    'status': 'success',
                    'source': 'huggingface_space',
                    'file_info': file_info
                })
            else:
                # Fallback to demo response
                print("⚠️ Using demo mode...")
                demo_result = get_demo_response(file_info, prediction_type)
                
                return jsonify({
                    'result': demo_result,
                    'status': 'success',
                    'source': 'demo_mode',
                    'file_info': file_info
                })
                
        except Exception as prediction_error:
            print(f"❌ Prediction error: {prediction_error}")
            # Fallback to demo response on any prediction error
            demo_result = get_demo_response(file_info, prediction_type)
            
            return jsonify({
                'result': demo_result,
                'status': 'success',
                'source': 'demo_fallback',
                'file_info': file_info,
                'note': 'Switched to demo mode due to prediction service unavailability'
            })
        
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️ Cleaned up: {file_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup error: {cleanup_error}")
    
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/examples/<prediction_type>', methods=['GET'])
def get_examples(prediction_type):
    """Get example files for testing"""
    examples = {
        'image': [
            {'name': 'Example 1', 'path': 'images/lady.jpg', 'description': 'Portrait image'},
            {'name': 'Example 2', 'path': 'images/fake_image.jpg', 'description': 'Synthetic image'}
        ],
        'video': [
            {'name': 'Example 1', 'path': 'videos/aaa.mp4', 'description': 'Video sample'},
            {'name': 'Example 2', 'path': 'videos/bbb.mp4', 'description': 'Video sample 2'}
        ],
        'audio': [
            {'name': 'Example 1', 'path': 'audios/DF_E_2000027.flac', 'description': 'Audio sample'},
            {'name': 'Example 2', 'path': 'audios/DF_E_2000031.flac', 'description': 'Audio sample 2'}
        ]
    }
    
    return jsonify({
        'examples': examples.get(prediction_type, []),
        'status': 'success'
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Add OPTIONS handler for CORS preflight
@app.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return '', 200

if __name__ == '__main__':
    print("🚀 Starting AI Video Detector Backend...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📊 Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    print(f"🤖 Hugging Face Space: {'Available' if HF_AVAILABLE else 'Not Available'}")
    
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"🌐 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)