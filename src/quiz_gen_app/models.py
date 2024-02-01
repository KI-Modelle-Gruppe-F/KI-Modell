# imports
import random
# import string
# import math
import torch
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, RobertaForQuestionAnswering, T5ForConditionalGeneration, AutoTokenizer, T5Tokenizer


# create Question class and load models + tokenizers


class Question:
    def __init__(self, question="", answer="", wrong_answers=None, answer_outlet=None):
        self.question = question
        self.answer = answer
        self.wrong_answers = wrong_answers if wrong_answers is not None else []
        self.answer_outlet = answer_outlet if answer_outlet is not None else []

    def __str__(self):
        return f"Question: {self.question}\nAnswer: {self.answer}\nWrong Answers: {self.wrong_answers}"

    def to_dict(self):
        return {"question": self.question, "answer": self.answer, "wrong_answers": self.wrong_answers}

    def from_dict(self, d):
        self.question = d["question"]
        self.answer = d["answer"]
        self.wrong_answers = d["wrong_answers"]

    def get_answer(self):
        return self.answer

    def get_question(self):
        return self.question

    def get_wrong_answers(self):
        return self.wrong_answers

    def set_answer(self, answer):
        self.answer = answer

    def set_question(self, question):
        self.question = question

    def get_wrong_answers(self):
        return self.wrong_answers

    def add_wrong_answers(self, wrong_answer):
        self.wrong_answers.append(wrong_answer)

    def remove_wrong_answers(self, wrong_answer):
        self.wrong_answers.remove(wrong_answer)

    def set_wrong_answers(self, wrong_answers):
        self.wrong_answers = wrong_answers

    def get_answer_outlet(self):
        return self.answer_outlet

    def get_answer_outlet_parts(self):
        return self.answer_outlet[0], self.answer_outlet[1], self.answer_outlet[2]

    def set_answer_outlet(self, answer_outlet):
        self.answer_outlet = answer_outlet

# create cached function to load models and tokenizers


@st.cache_resource(show_spinner=False)
def load_question_generation_model():
    # st.write('Loading question generation model...')
    gq_model_name = "thangved/t5-generate-question"  # 60.5M params
    ml = AutoModelForSeq2SeqLM.from_pretrained(gq_model_name)
    tr = AutoTokenizer.from_pretrained(gq_model_name)
    return ml, tr


@st.cache_resource(show_spinner=False)
def load_question_answering_model():
    # st.write('Loading question answering model...')
    # qa_model_name = 'deepset/roberta-large-squad2' # 354M params
    qa_model_name = 'deepset/tinyroberta-squad2'  # 81.5M params
    ml = RobertaForQuestionAnswering.from_pretrained(qa_model_name)
    tr = AutoTokenizer.from_pretrained(qa_model_name)
    return ml, tr


@st.cache_resource(show_spinner=False)
def load_wrong_answer_generation_model():
    # st.write('Loading wrong answer generation model...')
    gwa_model_name = "google/flan-t5-large"  # 783M params
    # gwa_model_name = "google/flan-t5-small"  # 77M params
    ml = T5ForConditionalGeneration.from_pretrained(gwa_model_name)
    tr = T5Tokenizer.from_pretrained(gwa_model_name)
    return ml, tr


def get_answer_outlet(p_question, num_wa):
    correct_answer_idx = random.randint(0, num_wa)
    answer_outlet = [q for q in p_question.get_wrong_answers()]
    answer_outlet.insert(correct_answer_idx, p_question.get_answer())
    return [p_question.get_question(), answer_outlet, correct_answer_idx]


def generate_questions(m, t, context, progress_bar):
    # progress start
    progress_text = "Generating questions..."
    progress_bar.progress(0, text=progress_text)

    gq_prompt = f"gq: {context}"

    # shorten text to fit in 512 tokens
    # msl = 512
    # gq_prompt = gq_prompt[:msl]

    input_ids = t(gq_prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        outputs = m.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=512,
            num_beams=3,
            no_repeat_ngram_size=5,
            early_stopping=True,
        )

    decoded_output = t.decode(
        outputs[0], skip_special_tokens=True).strip()
    # print(decoded_output)

    generated_questions = decoded_output.split('Question:')
    generated_questions = [Question(question=question.strip(
    )) for question in generated_questions if question.strip() != ""]
    # st.write('Questions generated')

    # progress finish
    # progress_value = 100/len(generated_questions)
    progress_value = 20
    progress_bar.progress(
        progress_value, text=f'{progress_text} {progress_value}%')

    return generated_questions


def generate_answers(m, t, context, generated_questions, progress_bar):
    # progress start
    progress_text = "Generating answers..."
    progress_bar.progress(20, text=progress_text)

    if len(generated_questions) == 0:
        raise Exception("No questions generated")

    questions = []

    for q_question in generated_questions:
        # st.write('an answer was generated')
        inputs = t(
            q_question.get_question(),
            context,
            return_tensors='pt',
            max_length=256,
            truncation=True
        )

        with torch.no_grad():
            outputs = m(**inputs, return_dict=True)

        # Find start and end indices with highest logits
        start_idx = torch.argmax(outputs['start_logits']).item()
        end_idx = torch.argmax(outputs['end_logits']).item() + 1

        answer = t.decode(
            inputs['input_ids'][0][start_idx:end_idx], skip_special_tokens=True)
        if answer.strip() != "":
            q_question.set_answer(answer.strip())
            questions.append(q_question)

    # progress finish
    # progress_value = 100/len(generated_questions)
    progress_value = 40
    progress_bar.progress(
        progress_value, text=f'{progress_text} {progress_value}%')

    return questions


def generate_wrong_answers(m, t, context, p_questions, num_wa, progress_bar):
    # Progress Start
    progress_text = "Finalizing Quiz..."
    q_len = len(p_questions)

    if q_len == 0:
        q_len = 1

    # (remaining progress / number of questions) / number of wrong answers
    p_incr = int((60 / q_len) / num_wa)

    progress = 40
    progress_bar.progress(progress, text=progress_text)

    for question in p_questions:
        # st.write(question)
        # gwa_prompt = f'Generate a wrong answer\nQuestion: {question.get_question()}\nCorrect Answer: {question.get_answer()}\nContext: {context}'

        wrong_answers = []

        exceptions = ''

        # if len(wrong_answers) > 0:
        #     wa = ', '.join(wrong_answers)
        #     exceptions = f'not being {wa}'

        gwa_prompt = f'Context:{context}\nQuestion: {question.get_question()}\nTask: Generate a wrong answer {exceptions}'

        for turn in range(num_wa):

            # st.write(f'{turn} wrong answer generated')
            input_ids = t(
                gwa_prompt, return_tensors="pt").input_ids

            with torch.no_grad():
                outputs = m.generate(
                    input_ids,
                    do_sample=True,
                    temperature=1.6,
                    no_repeat_ngram_size=5,
                    num_beams=1,
                    max_length=128,
                    # early_stopping=True,
                )

            decoded_output = t.decode(
                outputs[0], skip_special_tokens=True).strip()

            progress += p_incr

            # Capitalize first letter and make other letters lowercase
            output = decoded_output.capitalize(
            ) if question.get_answer()[0].isupper() else decoded_output

            # Capitalizes all words, but makes other letters lowercase
            # decoded_output = string.capwords(
            # decoded_output) if decoded_output[0].isupper() else decoded_output

            # Only Capitalizes first letter of each word
            # decoded_output = ' '.join(word[0].upper(
            # ) + word[1:] for word in decoded_output.split()) if decoded_output[0].isupper() else decoded_output

            # Onl capitalizes first letter of first word
            # decoded_output = f'{decoded_output[0].upper()}{decoded_output[1:]}' if decoded_output[0].isupper(
            # ) else decoded_output

            wrong_answers.append(output)
            progress_bar.progress(
                progress, text=f'{progress_text} {progress}%')

        question.set_wrong_answers(wrong_answers)
        question.set_answer_outlet(get_answer_outlet(question, num_wa))

    return p_questions


__all__ = ['Question', 'load_question_generation_model', 'load_question_answering_model',
           'load_wrong_answer_generation_model', 'generate_questions', 'generate_answers', 'generate_wrong_answers']
