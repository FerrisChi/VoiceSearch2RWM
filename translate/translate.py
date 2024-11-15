import time

from transformers import MarianMTModel, AutoTokenizer
from transformers import M2M100ForConditionalGeneration
from tokenization_small100 import SMALL100Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# opus
model_name = f'Helsinki-NLP/opus-mt-ja-en'
translate_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="pretrained_models/opus")
translate_model = MarianMTModel.from_pretrained(model_name, cache_dir="pretrained_models/opus")

# t5
# translate_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base", cache_dir="pretrained_models/t5")
# translate_model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base", cache_dir="pretrained_models/t5")

# Small100 model
# translate_model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100", cache_dir="pretrained_models/small100")
# translate_tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100", cache_dir="pretrained_models/small100")


def translate_to_english(text, lang='ja'):
    tim = time.time()
    # Small100: good but slow
    # encoded_text = translate_tokenizer(text, return_tensors="pt", padding=True)
    # generate_ids = translate_model.generate(**encoded_text)
    # translated_text = translate_tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

    # t5 bad
    # input_ids = translate_tokenizer("translate Japanese to English: 全都つながる素晴らしい方法です", return_tensors="pt").input_ids
    # print(input_ids)
    # outputs = translate_model.generate(input_ids)
    # translated_text = translate_tokenizer.decode(outputs[0], skip_special_tokens=True)


    # opus: moderate
    texts = [
        "全都つながる素晴らしい方法です。",
        "きゅぎの間を歩き振りのサイズリア、川のせせらぎを耳にしながら一歩一歩進んでいくと日常の"
    ]
    batch = translate_tokenizer(texts, return_tensors="pt", padding=True)
    generate_ids = translate_model.generate(**batch)
    translated_text = translate_tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
    tim = time.time() - tim
    print('Translation Time: {:.5f}'.format(tim))
    print(translated_text)
    
    return translated_text
