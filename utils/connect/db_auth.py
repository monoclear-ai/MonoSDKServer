from models import User
from utils.constants import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION
from util import time_log
from db_base import db_base
import boto3

# User database
# 1. name
# 2. email
# 3. eval key
# 4. (TODO) purchases
class db_auth(db_base):
    @classmethod
    def init(cls):
        cls.db = boto3.resource('dynamodb',
                                aws_access_key_id=AWS_ACCESS_KEY,
                                aws_secret_access_key=AWS_SECRET_KEY,
                                region_name=AWS_REGION)
        cls.table = cls.db.Table('mono-next-auth')
        return cls.table.creation_date_time

    @classmethod
    def get(cls, pk):
        response = cls.table.get_item(
            Key={'pk': pk, 'sk': pk}
        )
        try:
            return response['Item']
        except KeyError:
            return None