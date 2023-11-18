import json
from utils.connect.util import DecimalEncoder
from utils.connect.models import Eval
from utils.constants import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION
from utils.connect.util import time_log
from utils.connect.db_base import db_base
import boto3

# Eval database
# 1. Model tags per history
# 2. MMLU score history
# 3. FLASK score history
# 4. dbias score history
# 5. dtoxify score history
# 6. MMLU mistakes history (editable)
# 7. (TODO) Custom eval score history
# 8. (TODO) Custom eval mistakes history (editable)
class db_evals(db_base):
    @classmethod
    def init(cls):
        cls.db = boto3.resource('dynamodb',
                                aws_access_key_id=AWS_ACCESS_KEY,
                                aws_secret_access_key=AWS_SECRET_KEY,
                                region_name=AWS_REGION)
        cls.table = cls.db.Table('mono-evals')
        return cls.table.creation_date_time

    @classmethod
    def create(cls, eval_key, content):
        print(content)
        validate = Eval(**content)
        cls.table.put_item(
            Item={'eval_key': eval_key,
                  'pst': time_log.get_pst(), 'kst': time_log.get_kst(),
                  **content},
            ConditionExpression='attribute_not_exists(eval_key)'
        )

    @classmethod
    def update(cls, eval_key, update_dict):
        update_dict = json.loads(json.dumps(update_dict, cls=DecimalEncoder))
        # TODO : validate update_dict
        from decimal import Decimal
        exprs = [(k, v) for k, v in update_dict.items()]
        return cls.table.update_item(
            Key={'eval_key': eval_key},
            UpdateExpression='SET ' + ', '.join([f'{k} = :{k}' for k, v in exprs]),
            ExpressionAttributeValues={f':{k}': json.loads(json.dumps(v), parse_float=Decimal) for k, v in exprs},
            ReturnValues="UPDATED_NEW"
        )

    @classmethod
    def get(cls, eval_key):
        response = cls.table.get_item(
            Key={'eval_key': eval_key}
        )
        try:
            return response['Item']
        except KeyError:
            return None


