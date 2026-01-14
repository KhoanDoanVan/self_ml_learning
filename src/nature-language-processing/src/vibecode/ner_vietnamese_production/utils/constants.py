"""Constants for Vietnamese NER project."""

# Entity types
ENTITY_TYPES = {
    'PER': 'Person',           # Người
    'ORG': 'Organization',     # Tổ chức
    'LOC': 'Location',         # Địa điểm
    'MISC': 'Miscellaneous',   # Khác
    'DATE': 'Date',            # Ngày tháng
    'NUM': 'Number'            # Số
}

# BIO tagging scheme
BIO_TAGS = ['O', 'B', 'I']

# Special tokens
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"

# Label IDs
PAD_LABEL_ID = -100

# Vietnamese specific patterns
VIETNAMESE_STOP_WORDS = {
    'và', 'của', 'có', 'trong', 'được', 'với', 'là', 'để',
    'các', 'này', 'đã', 'cho', 'hay', 'từ', 'một', 'như'
}

# Regex patterns for Vietnamese entities
PATTERNS = {
    'PHONE': r'\b0\d{9,10}\b',
    'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'URL': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    'MONEY': r'\b\d+(?:[.,]\d+)?\s*(?:đồng|VND|USD|EUR)\b'
}

# Model parameters
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 5e-5

# File extensions
SUPPORTED_FILE_FORMATS = ['.txt', '.conll', '.json', '.csv']