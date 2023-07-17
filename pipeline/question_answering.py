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
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
qa_tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2").to(device)


def get_answer(question, text):  # Code taken from https://huggingface.co/transformers/task_summary.html
    inputs = qa_tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].tolist()[0]

    answer_start_scores, answer_end_scores = qa_model(**inputs, return_dict=False)

    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    ans = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return ans

def get_answer_batch(questions, texts):
    sample_num = len(questions)
    assert len(texts) == sample_num
    inputs = qa_tokenizer.batch_encode_plus([(questions[i], texts[i]) for i in range(sample_num)], add_special_tokens=True, padding=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].tolist()

    answer_start_scores, answer_end_scores = qa_model(**inputs, return_dict=False)

    answers = []
    for i in range(sample_num):
        answer_start = torch.argmax(
            answer_start_scores[i]
        )  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores[i]) + 1  # Get the most likely end of answer with the argmax of the score

        ans = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[i][answer_start:answer_end]))
        answers.append(ans)
    return answers

# model_name = "ktrapeznikov/albert-xlarge-v2-squad-v2"
#
# nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
#
#
# def get_answer(question, text):
#     QA_input = {
#         'question': question,
#         'context': text
#     }
#     res = nlp(QA_input, handle_impossible_answer=True)
#
#     return res['answer']


