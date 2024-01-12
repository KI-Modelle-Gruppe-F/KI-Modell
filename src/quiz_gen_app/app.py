import streamlit as st
import streamlit_book as stb

from models import generate_answers, generate_questions, generate_wrong_answers

# Initialize session state
# if 'questions' not in st.session_state:
#     st.session_state.questions = []
# if 'questions_array' not in st.session_state:
#     st.session_state.questions_array = []
if 'quiz' not in st.session_state:
    st.session_state.quiz = []

# Streamlit app
st.title("Quiz Generator")
st.sidebar.title("Quiz Generator Options")
num_wrong_answers = st.sidebar.slider("Number of wrong answers", 1, 5, 3)

# Input context
prompt_context = st.text_area("Enter the context:", height=300)

# Generate questions button
if st.button("Generate Questions"):
    # reset questions
    st.session_state.quiz = []

    # generate questions
    st.write("Generating questions...")
    questions = generate_questions(prompt_context)
    st.write("Generating answers...")
    questions_array = generate_answers(prompt_context, questions)
    st.write("Generating quiz...")
    st.session_state.quiz = generate_wrong_answers(
        questions_array, num_wrong_answers)


# Display quiz questions
for question in st.session_state.quiz:
    q, a, c = question.get_answer_outlet_parts()
    stb.single_choice(q, a, c)
