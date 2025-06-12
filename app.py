import os
import tempfile
import boto3
import cv2
import psycopg2
from openai import OpenAI
from PIL import Image
import io
import requests
import numpy as np
import json
import base64

from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

import insightface
from insightface.app import FaceAnalysis

# Configure OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def check_image_content_openai(image_path):
    """
    Use GPT-4 Vision to check if the image content is appropriate
    Returns True if the image is appropriate, False otherwise
    """
    try:
        # Read the image file and convert to base64
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            img_b64 = base64.b64encode(image_bytes).decode()

        # Create the prompt
        prompt = """
        Analyze this image and determine if it contains any of the following:
        1. Child sexual abuse material (CSAM)
        2. Visible exposure of extremely sensitive body parts, including genitals or nipples (e.g., full nudity or explicit close-ups)
        3. Extreme violence or gore
        
        Note: Proper sexual content is allowed.
        
        Respond with ONLY 'PASS' if the image is appropriate, or 'FAIL' if it contains any of the above.
        """

        try:
            # Generate content using GPT-4 Vision
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            # Log detailed response information
            print("OpenAI API Response Details:")
            print(f"- Response object type: {type(response)}")
            
            # Get the response text
            result = response.choices[0].message.content.strip().upper()
            print(f"OpenAI API Result: {result}")
            
            return result == 'PASS'
            
        except Exception as openai_error:
            # Log the specific error for debugging
            error_str = str(openai_error).lower()
            print(f"OpenAI API error details: {error_str}")
            print(f"Error type: {type(openai_error)}")
            
            # If the error contains content policy or safety-related terms, treat as FAIL
            if any(term in error_str for term in ['content', 'policy', 'safety']):
                print("Content blocked by safety filters - treating as FAIL")
                return False
                
            # For other errors, raise the exception to be handled by the outer try-catch
            raise
        
    except Exception as e:
        # For any other errors (like file reading issues), raise the exception
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

def send_discord_webhook(avatar_id, face_detection_pass, image_content_pass, check_pass, temp_img_path):
    """
    Send moderation results to Discord webhook
    """
    try:
        print("Starting Discord webhook function")
        print(f"Parameters received - temp_img_path: {temp_img_path}")
        
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        print(f"Discord webhook URL present: {bool(webhook_url)}")
        if not webhook_url:
            print("Discord webhook URL not found in environment variables")
            return
        
        # Create the webhook payload
        payload = {
            "embeds": [{
                "title": "Avatar Moderation Results",
                "color": 0x00ff00 if check_pass else 0xff0000,
                "fields": [
                    {
                        "name": "Avatar ID",
                        "value": str(avatar_id),
                        "inline": True
                    },
                    {
                        "name": "Face Detection",
                        "value": "✅ Pass" if face_detection_pass else "❌ Fail",
                        "inline": True
                    },
                    {
                        "name": "Content Check",
                        "value": "✅ Pass" if image_content_pass else "❌ Fail",
                        "inline": True
                    },
                    {
                        "name": "Overall Result",
                        "value": "✅ Pass" if check_pass else "❌ Fail",
                        "inline": True
                    }
                ],
                "image": {
                    "url": "attachment://image.jpg"
                }
            }]
        }

        print("Preparing to send webhook with image")
        # Prepare the multipart form data
        files = {
            'file': ('image.jpg', open(temp_img_path, 'rb'), 'image/jpeg'),
            'payload_json': (None, json.dumps(payload))
        }

        # Send the webhook with the image
        print("Sending webhook request...")
        response = requests.post(webhook_url, files=files)
        response.raise_for_status()
        print("Discord webhook sent successfully")

    except Exception as e:
        print(f"Error sending Discord webhook: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

def update_avatar_moderation(avatar_id, face_detection_result, image_content_result, temp_img_path=None):
    """
    Insert or update the moderation fields in avatar_moderation table
    """
    try:
        print(f"Attempting to update database for avatar_id: {avatar_id}")
        print(f"Face detection result: {face_detection_result}")
        print(f"Image content result: {image_content_result}")
        print(f"Webhook parameters - temp_img_path: {temp_img_path}")
        
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

        # Send Discord webhook if temp_img_path is provided
        if temp_img_path:
            print("Calling send_discord_webhook")
            send_discord_webhook(avatar_id, face_detection_result, image_content_result, check_pass, temp_img_path)
        else:
            print("Webhook not called because temp_img_path is missing")
        
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
    temp_img_path = None
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
                
                # Download the image first
                temp_img_path = download_from_s3(body["img_path"])
                if temp_img_path is None:
                    raise ValueError("Failed to download image")
                
                detect_face(body["img_path"], body["avatar_id"], is_local, temp_img_path)
                
            except json.JSONDecodeError as e:
                print(f"Error decoding message body: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing record: {str(e)}")
                continue
            finally:
                # Clean up temp file if it exists
                if temp_img_path and os.path.exists(temp_img_path):
                    os.unlink(temp_img_path)
                    print("Temporary file cleaned up in handler")
        
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Processing completed"})
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        # Clean up temp file if it exists
        if temp_img_path and os.path.exists(temp_img_path):
            os.unlink(temp_img_path)
            print("Temporary file cleaned up in handler error")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

def detect_face(img_path, avatar_id, is_local=False, temp_img_path=None):
    # Initialize face analysis with default settings for 0.2.1
    print(f"Starting face detection for avatar_id: {avatar_id}")
    app = FaceAnalysis(providers=['CPUExecutionProvider'], root='/tmp/models')
    
    try:
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
            
        # Check image content using OpenAI
        print("Starting image content check")
        image_content_result = check_image_content_openai(temp_img_path)
        print(f"Image content check result: {image_content_result}")
            
        # Only update database if it's not a local image
        if not is_local:
            print("Updating database for non-local image")
            update_avatar_moderation(avatar_id, face_detection_result, image_content_result, temp_img_path)
        else:
            print("Local image detected - skipping database update")
            print(f"Face detection result: {face_detection_result}")
            print(f"Content check result: {image_content_result}")
            
    except Exception as e:
        print(f"Error in face detection: {str(e)}")
        raise Exception(f"Error in face detection: {str(e)}")