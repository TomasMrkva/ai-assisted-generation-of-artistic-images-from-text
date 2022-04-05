import pprint as pp
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline
import yake
from keybert import KeyBERT
from multi_rake import Rake

kw_model = KeyBERT(model='all-mpnet-base-v2')
hf_name = 'pszemraj/led-large-book-summary'
summary_model = AutoModelForSeq2SeqLM.from_pretrained(
                hf_name,
                # low_cpu_mem_usage=True,
                )
summary_tokenizer = AutoTokenizer.from_pretrained(hf_name)   
have_GPU = torch.cuda.is_available()
summarizer = pipeline("summarization", model=summary_model, tokenizer=summary_tokenizer, device=0 if have_GPU else -1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
headline_model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
headline_tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")
headline_model = headline_model.to(device)

def make_headline(summary):
  text =  "headline: " + summary
  encoding = headline_tokenizer.encode_plus(text, return_tensors = "pt")
  input_ids = encoding["input_ids"].to(device)
  attention_masks = encoding["attention_mask"].to(device)
  bad_words_ids = headline_tokenizer(["Book", "-", "Story", "story", "Review", "Novel", "book", "novel", "review", "tale", "Tale", "A"], add_special_tokens=False).input_ids   

  beam_outputs = headline_model.generate(
      bad_words_ids=bad_words_ids,
      input_ids = input_ids,
      attention_mask = attention_masks,
      max_length = 30,
      num_beams = 4,
      # early_stopping = True,
      repetition_penalty = 1.0
  )

  return headline_tokenizer.decode(beam_outputs[0], skip_special_tokens=True)

def summarize_led(text):
  print('running')
  result = summarizer(
            text,
            min_length=16, 
            max_length=450,
            no_repeat_ngram_size=3, 
            encoder_no_repeat_ngram_size =3,
            clean_up_tokenization_spaces=True,
            repetition_penalty=3.7,
            num_beams=4,
            early_stopping=True,
      )
  return result[0]['summary_text']

def keyword_extractios(text, n):
  yake_kw = yake.KeywordExtractor(top=1, stopwords=None, n=n).extract_keywords(text)[0][0]
  rake_kw = Rake(max_words=n).apply(text)[0][0]
  bert_kw=kw_model.extract_keywords(text, keyphrase_ngram_range=(1, n), stop_words=None, top_n=1)[0][0]
  return yake_kw, rake_kw, bert_kw


def run(text):
    summary_led = summarize_led(text)
    headline = make_headline(summary_led)
    yake_kw, rake_kw, bert_kw = keyword_extractios(summary_led, n=8)
    return headline, yake_kw, rake_kw, bert_kw