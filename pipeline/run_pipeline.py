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

import argparse

import numpy as np
import pandas as pd
import spacy

import question_generation as qg
import question_answering as qa
from score import f1_score, clean_text
from tqdm import tqdm

INVALID_QUESTION = -1
NO_ANS = '[CLS]'
NO_VALID_QUESTIONS = 'NO_Q'

nlp = spacy.load("en_core_web_sm")


def filter_questions(exp_ans, pred_ans):
    if pred_ans == NO_ANS:
        return 'NO MATCH'
    if clean_text(exp_ans) != clean_text(pred_ans):
        return 'NO MATCH'
    return 'VALID'


def non_personal(question):
    question_tok = nlp(question)
    for tok in question_tok:
        if tok.dep_ == 'nsubj':
            if tok.text.lower() == 'i' or tok.text.lower() == 'you':
                return False
        elif tok.dep_ == 'poss':
            if tok.text.lower() == 'my' or tok.text.lower() == 'your':
                return False
    return True


def single_question_score(question, cand, response, knowledge):
    pred_ans = qa.get_answer(question, response)

    if filter_questions(cand, pred_ans) == 'VALID':
        knowledge_ans = qa.get_answer(question, knowledge)
        if knowledge_ans != NO_ANS:
            return f1_score(cand, knowledge_ans), knowledge_ans
        else:
            return 0, NO_ANS
    else:
        return INVALID_QUESTION, INVALID_QUESTION

def multiple_questions_score(questions, cands, responses, knowledges):
    pred_answers = qa.get_answer_batch(questions, responses)

    is_valid = []
    for i in range(len(questions)):
        cand = cands[i]
        pred_ans = pred_answers[i]
        if filter_questions(cand, pred_ans) == 'VALID':
            is_valid.append(True)
        else:
            is_valid.append(False)

    valid_questions = [questions[i] for i in range(len(questions)) if is_valid[i]]
    valid_knowledges = [knowledges[i] for i in range(len(questions)) if is_valid[i]]
    valid_ids = [i for i in range(len(questions)) if is_valid[i]]
    id_to_new_id = {valid_ids[i]: i for i in range(len(valid_ids))}
    knowledge_answers = qa.get_answer_batch(valid_questions, valid_knowledges)

    res = []
    for i in range(len(questions)):
        cand = cands[i]
        if is_valid[i]:
            knowledge_ans = knowledge_answers[id_to_new_id[i]]
            if knowledge_ans != NO_ANS:
                res.append((f1_score(cand, knowledge_ans), knowledge_ans))
            else:
                res.append((0, NO_ANS))
        else:
            res.append((INVALID_QUESTION, INVALID_QUESTION))

    return res

def get_response_score(response, knowledge, gen_method, single, remove_personal):
    f1 = 0
    num_questions = 0

    valid_questions = []
    valid_cands = []
    knowledge_answers = []
    scores = []

    candidates = qg.get_answer_candidates(response)
    for cand in candidates:
        if gen_method == 'greedy':
            questions = [qg.get_question_greedy(cand, response)]
        elif gen_method == 'beam':
            questions = qg.get_questions_beam(cand, response)
        else:
            questions = qg.get_questions_sample(cand, response)

        for question in questions:
            if not remove_personal or non_personal(question):
                question_score, knowledge_ans = single_question_score(question, cand, response, knowledge)
                if question_score != INVALID_QUESTION:
                    num_questions += 1
                    f1 += question_score

                    valid_questions.append(question)
                    valid_cands.append(cand)
                    knowledge_answers.append(knowledge_ans)
                    scores.append(question_score)

                    if single:
                        break
    if num_questions:
        avg_f1 = f1 / num_questions
    else:
        avg_f1 = INVALID_QUESTION
    return avg_f1, valid_questions, valid_cands, knowledge_answers, scores


def get_response_score_batch(responses, knowledges, gen_method, single, remove_personal, q_batch_size=50, a_batch_size=8):
    batch = {'cands': [], 'inds': []}
    entries = []
    sample_num = len(responses)
    assert sample_num == len(knowledges)
    for i in range(sample_num):
        response = responses[i]
        candidates = qg.get_answer_candidates(response)
        for j in range(len(candidates)):
            cand = candidates[j]
            batch['cands'].append(cand)
            batch['inds'].append(i)
            if len(batch['cands']) == q_batch_size or (i == sample_num - 1 and j == len(candidates) - 1):
                if gen_method == 'greedy':
                    #questions = [qg.get_question_greedy(cand, response)]
                    assert False # Not implemented
                elif gen_method == 'beam':
                    questions_lists = qg.get_questions_beam_batch(batch['cands'], [responses[k] for k in batch['inds']])
                    entries += [(batch['cands'][k], batch['inds'][k], questions_lists[k]) for k in range(len(batch['cands']))]
                else:
                    #questions = qg.get_questions_sample(cand, response)
                    assert False # Not implemented
                batch = {'cands': [], 'inds': []}

    # Re organize entires by original sample
    org_entries = []
    cur_org_entry = []
    cur_ind = 0
    for cand, ind, questions in entries:
        if cur_ind != ind:
            assert ind == cur_ind + 1
            cur_ind += 1
            org_entries.append(cur_org_entry)
            cur_org_entry = []

        cur_org_entry.append((cand, responses[ind], questions, knowledges[ind]))
    org_entries.append(cur_org_entry)

    batch = {'inds': []}
    ans_entries = {}
    for i in range(len(org_entries)):
        entry_list = org_entries[i]
        ans_entries[i] = {}
        for j in range(len(entry_list)):
            _, _, questions, _ = entry_list[j]
            ans_entries[i][j] = {}
            for k in range(len(questions)):
                question = questions[k]
                if not remove_personal or non_personal(question):
                    batch['inds'].append((i, j, k))
                if len(batch['inds']) == a_batch_size or (i == len(org_entries) - 1 and j == len(entry_list) - 1 and k == len(questions) - 1):
                    cur_res = multiple_questions_score(
                        [org_entries[batch['inds'][h][0]][batch['inds'][h][1]][2][batch['inds'][h][2]] for h in range(len(batch['inds']))],
                        [org_entries[batch['inds'][h][0]][batch['inds'][h][1]][0] for h in range(len(batch['inds']))],
                        [org_entries[batch['inds'][h][0]][batch['inds'][h][1]][1] for h in range(len(batch['inds']))],
                        [org_entries[batch['inds'][h][0]][batch['inds'][h][1]][3] for h in range(len(batch['inds']))]
                        )
                    for h in range(len(batch['inds'])):
                        ind1, ind2, ind3 = batch['inds'][h]
                        ans_entries[ind1][ind2][ind3] = cur_res[h]
                    batch = {'inds': []}

    res = []
    for i in range(len(org_entries)):
        entry_list = org_entries[i]
        for j in range(len(entry_list)):
            cand, response, questions, knowledge = entry_list[j]
            num_questions = 0
            f1 = 0
            valid_questions = []
            valid_cands = []
            knowledge_answers = []
            scores = []
            for k in range(len(questions)):
                if k in ans_entries[i][j]:
                    question_score, knowledge_ans = ans_entries[i][j][k]
                    if question_score != INVALID_QUESTION:
                        num_questions += 1
                        f1 += question_score

                        valid_questions.append(question)
                        valid_cands.append(cand)
                        knowledge_answers.append(knowledge_ans)
                        scores.append(question_score)

                        if single:
                            break
        if num_questions:
            avg_f1 = f1 / num_questions
        else:
            avg_f1 = INVALID_QUESTION
        res.append((avg_f1, valid_questions, valid_cands, knowledge_answers, scores))
    
    return res


def response_questions_stats(response, knowledge, gen_method, single, remove_personal):
    num_questions = 0
    num_no_ans = 0

    candidates = qg.get_answer_candidates(response)
    for cand in candidates:
        if gen_method == 'greedy':
            questions = [qg.get_question_greedy(cand, response)]
        elif gen_method == 'beam':
            questions = qg.get_questions_beam(cand, response)
        else:
            questions = qg.get_questions_sample(cand, response)

        for question in questions:
            if not remove_personal or non_personal(question):
                pred_ans = qa.get_answer(question, response)

                if filter_questions(cand, pred_ans) == 'VALID':
                    num_questions += 1
                    knowledge_ans = qa.get_answer(question, knowledge)
                    if knowledge_ans == NO_ANS:
                        num_no_ans += 1
                    if single:
                        break
    return num_questions, num_no_ans


def get_stats(in_path, gen_method, single, remove_personal):
    num_questions = 0
    num_no_ans = 0
    df = pd.read_csv(in_path)
    for _, row in df.iterrows():
        q, no_ans = response_questions_stats(row['response'], row['knowledge'], gen_method, single, remove_personal)
        num_questions += q
        num_no_ans += no_ans

    print("Total valid questions: {0}".format(num_questions))
    print("No answer: {0}".format(num_no_ans / num_questions))


def calc_scores(in_path, gen_method, single, remove_personal, out_path='', save_steps=False, q_batch_size=50, a_batch_size=8):
    print(in_path, gen_method, single, remove_personal)
    print(save_steps, flush=True)
    q_scores = []
    df = pd.read_csv(in_path)

    responses = []
    knowledges = []
    for idx, row in tqdm(df.iterrows()):
        responses.append(row['response'])
        knowledges.append(row['knowledge'])

    response_scores = get_response_score_batch(responses, knowledges, gen_method, single, remove_personal, q_batch_size, a_batch_size)

    all_questions = []
    all_cands = []
    all_answers = []
    all_scores = []
    all_responses = []
    all_knowledge = []
    ids = []

    for i in range(len(response_scores)):
        res, res_questions, res_cands, res_answers, res_scores = response_scores[i]
        all_questions.extend(res_questions)
        all_cands.extend(res_cands)
        all_answers.extend(res_answers)
        all_scores.extend(res_scores)
        all_responses.extend([responses[i]] * len(res_questions))
        all_knowledge.extend([knowledges[i]] * len(res_questions))
        ids.extend([idx] * len(res_questions))

        if res == INVALID_QUESTION:
            all_questions.extend([NO_VALID_QUESTIONS])
            all_cands.extend([NO_VALID_QUESTIONS])
            all_answers.extend([NO_VALID_QUESTIONS])
            all_scores.extend([INVALID_QUESTION])
            all_responses.extend([responses[i].lower()])
            all_knowledge.extend([knowledges[i]])
            ids.extend([idx])

        q_scores.append(res)

    if out_path != '':
        df['Q2'] = q_scores
        df = df[df.Q2 >= 0]
        df.to_csv(out_path + '.csv')

    if save_steps:
        data = {'id': ids, 'response': all_responses, 'cand': all_cands, 'question': all_questions, 'knowledge': all_knowledge,
                'knowledge_ans': all_answers, 'score': all_scores}
        steps_df = pd.DataFrame(data=data)
        steps_df.to_csv(out_path + '.steps.csv')

    valid_scores = [s for s in q_scores if s != -1]
    print("total with at least 1 valid question:", len(valid_scores))
    print("score:", np.mean(valid_scores))

    return valid_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True,
                        help="Path to a csv file containing dialogue model outputs.")
    parser.add_argument("--gen_method", type=str, required=True, choices=['greedy', 'beam', 'sampling'],
                        help="Decoding method for question generation.")
    parser.add_argument("--q_per_cand", type=str, choices=['single', 'multi'], default=['single'], required=False,
                        help="Take only one question per candidate when using beam/sampling for decoding")
    parser.add_argument("--personal", type=str, choices=['keep', 'remove'], default='keep', required=False,
                        help="Whether to remove personal questions.")
    parser.add_argument("--outfile", type=str, default='', required=False, help="Path to an output file")
    parser.add_argument("--save_steps", default=False, action="store_true", help="Whether to save all pipeline steps")
    parser.add_argument("--q_batch_size", type=int, default=50,
                        help="Size of batch for question generation model.")
    parser.add_argument("--a_batch_size", type=int, default=8,
                        help="Size of batch for question answering model.")
    args = parser.parse_args()

    if args.q_per_cand == 'single':
        single_q = True
    else:
        single_q = False

    if args.personal == 'remove':
        rm_personal = True
    else:
        rm_personal = False

    calc_scores(args.infile, args.gen_method, single=single_q, remove_personal=rm_personal,
                out_path=args.outfile, save_steps=args.save_steps, q_batch_size=args.q_batch_size,
                a_batch_size=args.a_batch_size)
