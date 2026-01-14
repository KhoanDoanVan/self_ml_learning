from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Liên quan vụ việc CSGT bị tố đánh dân, trúng một cháu nhỏ đang ngủ, đang lan truyền trên mạng xã hội, Đại tá Nguyễn Văn Tảo, Phó Giám đốc Công an tỉnh Tiền Giang vừa có cuộc họp cùng Chỉ huy Công an huyện Châu Thành và một số đơn vị nghiệp vụ cấp tỉnh để chỉ đạo làm rõ thông tin."
min_example = "Quán cơm bà Hương tốn 20 nghìn"
min_example2 = "đi coi phim Avatar 2 mất 35k"
ner_results = nlp(min_example2)
print(ner_results)

"""
[
    {
        "entity": "B-PERSON",
        "score": np.float32(0.99887997),
        "index": 28,
        "word": "Nguyễn",
        "start": 109,
        "end": 115
    },
    {
        "entity": "I-PERSON",
        "score": np.float32(0.99937624),
        "index": 29,
        "word": "Văn",
        "start": 116,
        "end": 119
    },
    {
        "entity": "I-PERSON",
        "score": np.float32(0.9993292),
        "index": 30,
        "word": "Tảo",
        "start": 120,
        "end": 123
    },
    {
        "entity": "B-ORGANIZATION",
        "score": np.float32(0.99909914),
        "index": 35,
        "word": "Công",
        "start": 138,
        "end": 142
    },
    {
        "entity": "I-ORGANIZATION",
        "score": np.float32(0.99949706),
        "index": 36,
        "word": "an",
        "start": 143,
        "end": 145
    },
    {
        "entity": "I-ORGANIZATION",
        "score": np.float32(0.99938357),
        "index": 37,
        "word": "tỉnh",
        "start": 146,
        "end": 150
    },
    {
        "entity": "I-ORGANIZATION",
        "score": np.float32(0.9992514),
        "index": 38,
        "word": "Tiền",
        "start": 151,
        "end": 155
    },
    {
        "entity": "I-ORGANIZATION",
        "score": np.float32(0.9992137),
        "index": 39,
        "word": "Giang",
        "start": 156,
        "end": 161
    },
    {
        "entity": "B-ORGANIZATION",
        "score": np.float32(0.99908066),
        "index": 47,
        "word": "Công",
        "start": 191,
        "end": 195
    },
    {
        "entity": "I-ORGANIZATION",
        "score": np.float32(0.9994672),
        "index": 48,
        "word": "an",
        "start": 196,
        "end": 198
    },
    {
        "entity": "I-ORGANIZATION",
        "score": np.float32(0.9990018),
        "index": 49,
        "word": "huyện",
        "start": 199,
        "end": 204
    },
    {
        "entity": "I-ORGANIZATION",
        "score": np.float32(0.99900836),
        "index": 50,
        "word": "Châu",
        "start": 205,
        "end": 209
    },
    {
        "entity": "I-ORGANIZATION",
        "score": np.float32(0.9988128),
        "index": 51,
        "word": "Thành",
        "start": 210,
        "end": 215
    }
]
"""

"""
min_example

[
    {
        'entity': 'B-PERSON',
        'score': np.float32(0.99538773),
        'index': 4,
        'word': 'Hương',
        'start': 12,
        'end': 17
    }
]

"""