from enum import Enum


class TextType(Enum):
    ARTICLE = 1
    COMMENT = 0

    @classmethod
    def get_type(cls, text):
        if text == "article":
            return cls.ARTICLE
        elif text == "comment":
            return cls.COMMENT
        return None

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)
