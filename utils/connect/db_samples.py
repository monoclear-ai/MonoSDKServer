import json
from utils.connect.util import DecimalEncoder
from utils.connect.models import Samples
from utils.constants import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION
from utils.connect.util import time_log
from utils.connect.db_base import db_base
import boto3


# TODO: Pagination
class db_samples(db_base):
    @classmethod
    def init(cls):
        cls.db = boto3.resource('dynamodb',
                                aws_access_key_id=AWS_ACCESS_KEY,
                                aws_secret_access_key=AWS_SECRET_KEY,
                                region_name=AWS_REGION)
        cls.table = cls.db.Table('mono-samples')
        return cls.table.creation_date_time

    @classmethod
    def create(cls, sample_key, content):
        print(content)
        validate = Samples(**content)
        cls.table.put_item(
            Item={'sample_key': sample_key,
                  'pst': time_log.get_pst(), 'kst': time_log.get_kst(),
                  **content},
            ConditionExpression='attribute_not_exists(sample_key)'
        )

    @classmethod
    def update(cls, sample_key, update_dict):
        update_dict = json.loads(json.dumps(update_dict, cls=DecimalEncoder))
        # TODO : validate update_dict
        from decimal import Decimal
        exprs = [(k, v) for k, v in update_dict.items()]
        return cls.table.update_item(
            Key={'sample_key': sample_key},
            UpdateExpression='SET ' + ', '.join([f'{k} = :{k}' for k, v in exprs]),
            ExpressionAttributeValues={f':{k}': json.loads(json.dumps(v), parse_float=Decimal) for k, v in exprs},
            ReturnValues="UPDATED_NEW"
        )

    @classmethod
    def get(cls, sample_key):
        response = cls.table.get_item(
            Key={'sample_key': sample_key}
        )
        try:
            return response['Item']
        except KeyError:
            return None
