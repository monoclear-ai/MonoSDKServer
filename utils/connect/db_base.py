from abc import ABC

class db_base(ABC):
    @classmethod
    def init(cls):
        pass

    @classmethod
    def create(cls, id, content):
        pass

    @classmethod
    def update(cls, id, content):
        pass

    @classmethod
    def get(cls, id):
        pass