from datetime import date, datetime
import decimal
import json
from typing import Iterable
from pytz import timezone
import pytz


class time_log:

    @classmethod
    def _get_utc(cls):
        date = datetime.now(tz=pytz.utc)
        return date

    @classmethod
    def get_pst(cls):
        utc = cls._get_utc()
        date = utc.astimezone(timezone('US/Pacific'))
        return date.isoformat()

    @classmethod
    def get_kst(cls):
        utc = cls._get_utc()
        date = utc.astimezone(timezone('Asia/Seoul'))
        return date.isoformat()


class tagger:
    @classmethod
    def unique_tag(cls, eval_key, tag):
        return f'{eval_key}__{tag}'

    @classmethod
    def display_tag(cls, unique_tag):
        return unique_tag.split('__', 1)[1]


class dict_serialize:
    @classmethod
    def serialize(cls, d):
        return ','.join([f"{k}={v}" for k, v in d.items()])


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()

        return json.JSONEncoder.default(self, o)


class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            # wanted a simple yield str(o) in the next line,
            # but that would mean a yield on the line with super(...),
            # which wouldn't work (see my comment below), so...
            return float(o)
        if isinstance(o, Iterable):
            return list(o)
        return super(DecimalEncoder, self).default(o)