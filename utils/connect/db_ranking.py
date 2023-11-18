import json
from utils.connect.util import DecimalEncoder
from utils.connect.models import Ranking
from utils.connect.db_base import db_base
from utils.constants import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION
from utils.connect.util import time_log
import boto3


class db_ranking(db_base):
    @classmethod
    def init(cls):
        cls.db = boto3.resource('dynamodb',
                                aws_access_key_id=AWS_ACCESS_KEY,
                                aws_secret_access_key=AWS_SECRET_KEY,
                                region_name=AWS_REGION)
        cls.table = cls.db.Table('mono-ranking')
        return cls.table.creation_date_time

    @classmethod
    def create(cls, task_privacy, content):
        validate = Ranking(**content)
        cls.table.put_item(
            Item={'task_privacy': task_privacy,
                  'pst': time_log.get_pst(), 'kst': time_log.get_kst(),
                  **content},
            ConditionExpression='attribute_not_exists(task_privacy)'
        )

    @classmethod
    def update(cls, task_privacy, update_dict):
        # TODO : validate update_dict
        update_dict = json.loads(json.dumps(update_dict, cls=DecimalEncoder))
        from decimal import Decimal
        exprs = [(k, v) for k, v in update_dict.items()]
        return cls.table.update_item(
            Key={'task_privacy': task_privacy},
            UpdateExpression='SET ' + ', '.join([f'{k} = :{k}' for k, v in exprs]),
            ExpressionAttributeValues={f':{k}': json.loads(json.dumps(v), parse_float=Decimal) for k, v in exprs},
            ReturnValues="UPDATED_NEW"
        )

    @classmethod
    def get(cls, task_privacy):
        response = cls.table.get_item(
            Key={'task_privacy': task_privacy}
        )
        try:
            return response['Item']
        except KeyError:
            return None
