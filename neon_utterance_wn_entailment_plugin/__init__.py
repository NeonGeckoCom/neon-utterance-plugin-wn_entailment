# # NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# # All trademark and other rights reserved by their respective owners
# # Copyright 2008-2021 Neongecko.com Inc.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS;  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import nltk
from nltk.corpus import wordnet as wn

from neon_transformers import UtteranceTransformer
from neon_transformers.tasks import UtteranceTask


class WordNetEntailments(UtteranceTransformer):
    task = UtteranceTask.TOPIC_EXTRACTION

    def __init__(self, name="WordNetEntailments", priority=99):
        super().__init__(name, priority)
        nltk.download("wordnet")

    def get_keywords(self, context):
        # NOTE: you need to install one of the
        # neon transformers that provides keywords
        # priority if this plugin defaults to 99
        # to ensure keyword extractors run first

        # keybert
        kws = context.get("keybert_keywords") or []
        # yake
        kws += context.get("yake_keywords") or []
        # rake
        kws += context.get("rake_keywords") or []

        if kws:
            return [k[0] for k in kws]
        return []

    def get_parent_verbs(self, keywords):
        parents = []
        for kw in keywords:
            sym = wn.synsets(kw)[:2]
            parents += [j.name().split('.')
                        for i in sym for j in i.hypernyms()]
        parents = [i[0] for i in parents if i[1] == 'v']
        parents = [i for i in parents if i not in keywords]
        return list(set(parents))

    def get_entailments(self, keywords):
        parents = []
        for kw in keywords:
            sym = wn.synsets(kw)[:2]
            parents += [j.name().split('.')
                        for i in sym for j in i.entailments()]
        parents = [i[0] for i in parents if i[1] == 'v']
        parents = [i for i in parents if i not in keywords]
        return list(set(parents))

    def get_labels(self, keywords):
        classes = []
        entailments = []
        for k in keywords:
            classes += self.get_parent_verbs([k.replace(" ", "_")])
            classes += self.get_parent_verbs(k.split())
        for k in keywords:
            entailments += self.get_entailments([k.replace(" ", "_")])
            entailments += self.get_entailments(k.split())
        for k in list(set(classes)):
            entailments += self.get_entailments([k])
        return list(set(entailments))

    def transform(self, utterances, context=None):
        context = context or {}
        keywords = self.get_keywords(context)
        classes = self.get_labels(keywords)

        # return unchanged utterances + data
        return utterances, {"entailments": classes}

