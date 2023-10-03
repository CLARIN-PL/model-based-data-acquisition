import re
import string
from abc import ABC


class TextFilter(ABC):
    def __init__(self, has_enough_words_threshold=4) -> None:
        self.enough_words_threshold = has_enough_words_threshold
        self.load_alphabet()

    def split_text_to_words(self, text):
        return text.split(" ")

    def load_alphabet(self):
        chars = "AaĄąBbCcĆćDdEeĘęFfGgHhIiJjKkLlŁłMmNnŃńOoÓóPpQqRrSsŚśTtUuVvWwXxYyZzŹźŻż"
        self.alphabet = re.compile(f"^[0-9{chars}\s{string.punctuation}„”–…’—““‘·”`]+$")
        return self.alphabet

    def is_in_alphabet(self, text):
        return bool(re.search(self.alphabet, text))

    def has_enough_words(self, text):
        length = len(self.split_text_to_words(text))
        return length > self.enough_words_threshold

    @property
    def filters(self):
        raise NotImplementedError()

    def is_text_valid(self, text):
        string_ = str(text)
        for filt in self.filters:
            if not filt(string_):
                return False
        return True

    def filter_texts(self, df):
        return df[df["text"].progress_apply(self.is_text_valid)]


class ParagraphFilter(TextFilter):
    def __init__(self, has_enough_words_threshold=5) -> None:
        super().__init__(has_enough_words_threshold)
        self.forbidden_word_list = ["zobacz", "poznaj"]
        self.forbidden_link_list = ["//"]

    def has_too_many_colons(self, text):
        shorter = text.replace(":", "")
        return len(shorter) + 1 >= len(text)

    def not_contain_forbidden_words(self, text):
        for word in self.split_text_to_words(text):
            if word.lower() in self.forbidden_word_list:
                return False
        return True

    def not_contain_forbidden_link(self, text):
        for word in self.split_text_to_words(text):
            for pattern in self.forbidden_link_list:
                if pattern in word:
                    return False
        return True

    @property
    def filters(self):
        return [
            self.is_in_alphabet,
            self.has_too_many_colons,
            self.has_enough_words,
            self.not_contain_forbidden_words,
            self.not_contain_forbidden_link,
        ]


class CommentFilter(TextFilter):
    def __init__(
        self,
        has_enough_words_threshold=4,
        has_enough_long_words_count=3,
        has_enough_long_words_length=4,
        unique_words_percent=0.5,
        same_char_words_percent=0.5,
    ) -> None:
        super().__init__(has_enough_words_threshold)
        self.has_enough_long_words_count = has_enough_long_words_count
        self.has_enough_long_words_length = has_enough_long_words_length
        self.unique_words_percent = unique_words_percent
        self.same_char_words_percent = same_char_words_percent

    def is_not_all_haha(self, text):
        haha = "ha"
        for word in self.split_text_to_words(text):
            if haha not in word:
                return True
        return False

    def has_enough_long_words(self, text):
        count = 0
        for word in self.split_text_to_words(text):
            if len(word) >= self.has_enough_long_words_length:
                count += 1
            if count >= self.has_enough_long_words_count:
                return True
        return False

    def has_not_many_same_char_words(self, text):
        splitted = self.split_text_to_words(text)
        size = len(splitted)
        chars_counter = {1: 0, 2: 0}
        for word in splitted:
            uniq_chars = len(set(word))
            if uniq_chars not in chars_counter:
                chars_counter[uniq_chars] = 0
            chars_counter[uniq_chars] += 1
        if chars_counter[1] + chars_counter[2] >= size * self.same_char_words_percent:
            return False
        return True

    def has_enough_unique_words(self, text):
        splitted = self.split_text_to_words(text)
        size = len(splitted)
        word_counter = {}
        for word in splitted:
            if word not in word_counter:
                word_counter[word] = 0
            word_counter[word] += 1
        for counter in word_counter.values():
            if counter >= size * self.unique_words_percent:
                return False
        return True

    @property
    def filters(self):
        return [
            self.is_in_alphabet,
            self.has_enough_words,
            self.is_not_all_haha,
            self.has_enough_long_words,
            self.has_not_many_same_char_words,
            self.has_enough_unique_words,
        ]
