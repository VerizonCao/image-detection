import os
import tempfile
import boto3
import cv2
import psycopg2
import google.generativeai as genai
from PIL import Image
import io
import requests
import numpy as np
import json

from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

import insightface
from insightface.app import FaceAnalysis

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


def check_image_content(image_path):
    """
    Use Gemini to check if the image content is appropriate
    Returns True if the image is appropriate, False otherwise
    """
    try:
        # Read the image file and convert to PIL Image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            pil_image = Image.open(io.BytesIO(image_bytes))

        # Create the model
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Create the prompt
        prompt = """
        Analyze this image and determine if it contains any of the following:
        1. Child sexual abuse material (CSAM)
        2. Visible exposure of sensitive body parts, including genitals or nipples (e.g., full nudity or explicit close-ups)
        3. Non-consensual sexual content
        4. Extreme violence or gore
        
        Note: Proper sexual content is allowed.
        
        Respond with ONLY 'PASS' if the image is appropriate, or 'FAIL' if it contains any of the above.
        """

        try:
            # Generate content
            response = model.generate_content([prompt, pil_image])
            
            # Check the response
            result = response.text.strip().upper()
            return result == 'PASS'
            
        except Exception as gemini_error:
            # If Gemini rejects the image due to policy, treat it as a FAIL
            error_str = str(gemini_error).lower()
            if any(keyword in error_str for keyword in [
                'policy', 'rejected', 'not allowed', 'inappropriate',
                'safety', 'content policy', 'violation'
            ]):
                print(f"Gemini policy rejection: {error_str}")
                return False
            # For other errors, re-raise
            raise
        
    except Exception as e:
        raise Exception(f"Error in image content check: {str(e)}")

def get_db_connection():
    """
    Create a connection to the Neon database
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv('PGHOST'),
            database=os.getenv('PGDATABASE'),
            user=os.getenv('PGUSER'),
            password=os.getenv('PGPASSWORD')
        )
        return conn
    except Exception as e:
        raise Exception(f"Error connecting to database: {str(e)}")

def update_avatar_moderation(avatar_id, face_detection_result, image_content_result):
    """
    Insert or update the moderation fields in avatar_moderation table
    """
    try:
        print(f"Attempting to update database for avatar_id: {avatar_id}")
        print(f"Face detection result: {face_detection_result}")
        print(f"Image content result: {image_content_result}")
        
        conn = get_db_connection()
        print("Database connection established")
        
        cur = conn.cursor()
        
        # Calculate check_pass based on both results
        check_pass = face_detection_result and image_content_result
        print(f"Calculated check_pass: {check_pass}")
        
        # Insert if not exists, update if exists
        cur.execute(
            """
            INSERT INTO avatar_moderation 
                (avatar_id, face_detection_pass, image_content_pass, check_pass)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (avatar_id) 
            DO UPDATE SET 
                face_detection_pass = EXCLUDED.face_detection_pass,
                image_content_pass = EXCLUDED.image_content_pass,
                check_pass = EXCLUDED.check_pass
            """,
            (avatar_id, face_detection_result, image_content_result, check_pass)
        )
        print("SQL query executed successfully")
        
        conn.commit()
        print("Database changes committed")
        
        cur.close()
        conn.close()
        print("Database connection closed")
        
    except Exception as e:
        print(f"Database error details: {str(e)}")
        raise Exception(f"Error updating avatar moderation: {str(e)}")

def generate_presigned_url(img_path):
    """
    Generate a presigned URL for the S3 object
    """
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name=os.getenv('AWS_REGION', 'us-west-2')
        )
        
        bucket_name = os.getenv('S3_BUCKET_NAME')
        
        # Generate presigned URL
        presigned_url = s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': bucket_name,
                'Key': img_path
            },
            ExpiresIn=300  # URL expires in 5 minutes
        )
        
        return presigned_url
    except Exception as e:
        raise Exception(f"Error generating presigned URL: {str(e)}")

def download_from_s3(img_path):
    """
    Download an image from S3 bucket to a temporary file or use local file
    Returns the path to the image file
    """
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_path = temp_file.name
        
        # If the path doesn't start with 'rita-avatars/', treat it as a local file
        if not img_path.startswith('rita-avatars/'):
            # Copy the local file to temp location
            import shutil
            shutil.copy2(img_path, temp_path)
            return temp_path
            
        # Generate presigned URL
        presigned_url = generate_presigned_url(img_path)
        print(f"Generated presigned URL for {img_path}")
        
        # Download from presigned URL
        try:
            response = requests.get(presigned_url)
            response.raise_for_status()
            
            # Save the image to temp file
            with open(temp_path, 'wb') as f:
                f.write(response.content)
                
            print(f"Successfully downloaded to {temp_path}")
            return temp_path
            
        except Exception as download_error:
            print(f"Error downloading file: {str(download_error)}")
            raise
            
    except Exception as e:
        raise Exception(f"Error downloading image: {str(e)}")

def handler(event, context):
    try:
        # Handle SQS event
        if not event.get('Records'):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid event format - no Records found"})
            }
            
        # Process each record (though typically SQS triggers one message at a time)
        for record in event['Records']:
            try:
                # Parse the message body
                body = json.loads(record['body'])
                
                # Check if required fields are present
                if not body.get('img_path'):
                    print("Missing required field: img_path")
                    continue
                    
                if not body.get('avatar_id'):
                    print("Missing required field: avatar_id")
                    continue
                    
                # Check if it's a local image (doesn't start with rita-avatars/)
                is_local = not body["img_path"].startswith('rita-avatars/')
                print(f"the img_path is {body['img_path']}, is_local: {is_local}")
                detect_face(body["img_path"], body["avatar_id"], is_local)
                
            except json.JSONDecodeError as e:
                print(f"Error decoding message body: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing record: {str(e)}")
                continue
        
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Processing completed"})
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

def detect_face(img_path, avatar_id, is_local=False):
    # Initialize face analysis with default settings for 0.2.1
    print(f"Starting face detection for avatar_id: {avatar_id}")
    app = FaceAnalysis(providers=['CPUExecutionProvider'], root='/tmp/models')
    
    try:
        # Download the image from S3 to temp file or use local file
        temp_img_path = download_from_s3(img_path)
        
        # load img
        if temp_img_path is not None:
            img_bgr = cv2.imread(temp_img_path, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Failed to download image from S3")

        print(img_bgr.shape)
        # adjust det_thresh, liveportrait used 0.5. Given model is deterministic, so we can use 0.5
        # anime style often lies between 0.05 to 0.1 det values
        app.prepare(ctx_id=0, det_thresh=0.5, det_size=(512, 512))
        faces = app.get(img_bgr)  # detected faces, list of faces

        print(len(faces))
        # if detection is successful, len(faces) > 0,
        face_detection_result = len(faces) > 0
        if face_detection_result:  # print the first score, which is detected with the biggest bbox
            print(f"{faces[0].det_score:.4f}")
            
        # Check image content using Gemini
        print("Starting image content check")
        image_content_result = check_image_content(temp_img_path)
        print(f"Image content check result: {image_content_result}")
            
        # Only update database if it's not a local image
        if not is_local:
            print("Updating database for non-local image")
            update_avatar_moderation(avatar_id, face_detection_result, image_content_result)
        else:
            print("Local image detected - skipping database update")
            print(f"Face detection result: {face_detection_result}")
            print(f"Content check result: {image_content_result}")
            
        # Clean up the temporary file
        os.unlink(temp_img_path)
        print("Temporary file cleaned up")
        
    except Exception as e:
        print(f"Error in face detection: {str(e)}")
        raise Exception(f"Error in face detection: {str(e)}")