import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Access the API key
apikey = os.getenv("APIKEY")

print("API Key:", apikey)
