from app import handler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_handler():
    # Test cases
    test_cases = [
        # {
        #     "name": "S3 Image Test",
        #     "event": {
        #         "img_path": "rita-avatars/Jim-1748192809448.png",  # S3 path
        #         "avatar_id": "a-gjlqfls9bs2"
        #     }
        # },
        {
            "name": "Local Image Test",
            "event": {
                "img_path": "/mnt/d/download/audio_live/f1.png",  # Local path
                "avatar_id": "test_local_123"
            }
        }
    ]

    # Test context (can be None for local testing)
    test_context = None

    # Run each test case
    for test_case in test_cases:
        print(f"\nRunning test: {test_case['name']}")
        try:
            # Call the handler
            response = handler(test_case["event"], test_context)
            print("Response:", response)
            print(f"Test '{test_case['name']}' completed successfully!")
        except Exception as e:
            print(f"Error during test '{test_case['name']}':", str(e))

if __name__ == "__main__":
    # Verify environment variables
    required_env_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_REGION',
        'S3_BUCKET_NAME',
        'PGHOST',
        'PGDATABASE',
        'PGUSER',
        'PGPASSWORD',
        'GOOGLE_API_KEY'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print("Missing environment variables:", missing_vars)
        print("Please set all required environment variables in your .env file")
    else:
        test_handler() 