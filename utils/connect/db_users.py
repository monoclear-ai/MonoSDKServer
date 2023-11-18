from utils.connect.models import User
from utils.constants import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION
from utils.connect.util import time_log
from utils.connect.db_base import db_base
import boto3

# User database
# 1. name
# 2. email
# 3. eval key
# 4. (TODO) purchases
class db_users(db_base):
    @classmethod
    def init(cls):
        cls.db = boto3.resource('dynamodb',
                                aws_access_key_id=AWS_ACCESS_KEY,
                                aws_secret_access_key=AWS_SECRET_KEY,
                                region_name=AWS_REGION)
        cls.table = cls.db.Table('mono-user')
        return cls.table.creation_date_time

    @classmethod
    def create(cls, id, content):
        validate = User(**content)
        cls.table.put_item(
            Item={'id': id,
                  'pst': time_log.get_pst(), 'kst': time_log.get_kst(),
                  **content},
            ConditionExpression='attribute_not_exists(id)'
        )

    @classmethod
    def update(cls, id, update_dict):
        cls.table.update_item(
            Item={'id': id,
                  **update_dict}
        )

    @classmethod
    def get(cls, id):
        response = cls.table.get_item(
            Key={'id': id}
        )
        try:
            return response['Item']
        except KeyError:
            return None