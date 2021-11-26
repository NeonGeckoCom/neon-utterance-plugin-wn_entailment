```python
from neon_utterance_KeyBERT_plugin import KeyBERTExtractor
from neon_utterance_wn_entailment_plugin import WordNetEntailments

from neon_utterance_KeyBERT_plugin import KeyBERTExtractor


kbert = KeyBERTExtractor()  # or RAKE or YAKE or all of them!

tars = WordNetEntailments()

utts = ["The man was snoring very loudly"]

_, context = kbert.transform(utts)
# {'keybert_keywords': [('snoring', 0.829), ('man snoring loudly', 0.8174), ('man snoring', 0.668)]}

_, context = tars.transform(utts, context)

print(context)
# {'entailments': ['exhale', 'inhale', 'sleep']}
```