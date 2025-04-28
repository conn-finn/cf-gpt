import re
import unidecode

class Tokenizer:
    def __init__(self):
        pass

    def tokenize(line):
        line = re.sub(r'[^a-zA-Z0-9]', ' ', unidecode.unidecode(line)) # remove punctuation
        line = line.lower().split()  # lower case
        return line