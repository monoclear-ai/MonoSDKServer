import os
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('../.env')
load_dotenv(dotenv_path=dotenv_path)

AWS_ACCESS_KEY = os.getenv('SERVER_AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('SERVER_AWS_SECRET_KEY')
AWS_REGION = os.getenv('SERVER_AWS_REGION')