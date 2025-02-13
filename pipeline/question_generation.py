# Copyright 2020 The Q2 Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForQuestionAnswering

import spacy


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
qg_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
qg_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap").to(device)

nlp = spacy.load("en_core_web_sm")


def get_answer_candidates(text):
    doc = nlp(text)
    candidates = [ent.text for ent in list(doc.ents)]
    noun_chunks = list(doc.noun_chunks)
    for chunk in noun_chunks:
        found = False
        for cand in candidates:
            if chunk.text.lower() == cand.lower():
                found = True
        if not found:
            candidates.append(chunk.text)
    # candidates += [chunk.text for chunk in list(doc.noun_chunks) if chunk.text not in candidates]
    candidates = [cand for cand in candidates if cand.lower() != 'i']
    return candidates


# def get_answer_candidates(text):
#     doc = nlp(text)
#     candidates = [ent.text for ent in list(doc.ents)]
#     candidates_lower = [c.lower() for c in candidates]
#     noun_chunks = list(doc.noun_chunks)
#     candidates += [c.text for c in noun_chunks if c.text.lower() not in candidates_lower and c.text.lower() != 'i']
#     return candidates


def get_question_greedy(answer, context, max_length=128):
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = qg_tokenizer([input_text], return_tensors='pt').to(device)

    output = qg_model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'],
                               max_length=max_length)

    question = qg_tokenizer.decode(output[0]).replace("question: ", "", 1)
    return question


def get_questions_beam(answer, context, max_length=128, beam_size=5, num_return=5):
    all_questions = []
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = qg_tokenizer([input_text], return_tensors='pt').to(device)

    beam_outputs = qg_model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'],
                                     max_length=max_length, num_beams=beam_size, no_repeat_ngram_size=3,
                                     num_return_sequences=num_return, early_stopping=True)

    for beam_output in beam_outputs:
        all_questions.append(qg_tokenizer.decode(beam_output, skip_special_tokens=True).replace("question: ", "", 1))

    return all_questions

def get_questions_beam_batch(answers, contexts, max_length=128, beam_size=5, num_return=5):
    batch_size = len(answers)
    assert batch_size == len(contexts)
    input_texts = ["answer: %s  context: %s </s>" % (answers[i], contexts[i]) for i in range(batch_size)]
    features = qg_tokenizer(input_texts, return_tensors='pt', padding=True).to(device)

    beam_outputs = qg_model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'],
                                     max_length=max_length, num_beams=beam_size, no_repeat_ngram_size=3,
                                     num_return_sequences=num_return, early_stopping=True)

    res = []
    cur_ind = 0
    beam_outputs_num = beam_outputs.shape[0]
    assert beam_outputs_num == beam_size*batch_size
    while cur_ind < beam_outputs_num:
        all_questions = []
        for offset in range(beam_size):
            beam_output = beam_outputs[cur_ind+offset]
            all_questions.append(qg_tokenizer.decode(beam_output, skip_special_tokens=True).replace("question: ", "", 1))
        res.append(all_questions)
        cur_ind = cur_ind + beam_size

    return res


def get_questions_sample(answer, context, max_length=128, top_k=50, top_p=0.95, num_return=5):
    all_questions = []
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = qg_tokenizer([input_text], return_tensors='pt').to(device)

    sampled_outputs = qg_model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'],
                                        max_length=max_length, do_sample=True, top_k=top_k, top_p=top_p,
                                        num_return_sequences=num_return)

    for sampled in sampled_outputs:
        all_questions.append(qg_tokenizer.decode(sampled, skip_special_tokens=True).replace("question: ", "", 1))

    return all_questions
