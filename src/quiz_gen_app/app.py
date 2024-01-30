from models import generate_answers, generate_questions, generate_wrong_answers
import time
import streamlit as st
import streamlit_book as stb

# Start of App Notice
from app_start import app_start
app_start()

# Streamlit app
# Initialize session state
if 'quiz' not in st.session_state:
    st.session_state.quiz = []
if 'current_state' not in st.session_state:
    st.session_state.current_state = 'input'
if 'prompt' not in st.session_state:
    st.session_state.prompt = ''
if 'num_wrong_answers' not in st.session_state:
    st.session_state.num_wrong_answers = 3


def toggle_state(state):
    st.session_state.current_state = state


# Input text state
if st.session_state.current_state == 'input':

    st.markdown('''Create a simple and short quiz from a given text! You can choose the number of wrong answers per question from the Sidebar.''')

    prompt_context = st.text_area(
        "Enter the Text:", value=st.session_state.prompt, height=300)
    st.session_state.num_wrong_answers = st.sidebar.slider(
        "Number of wrong answers", 1, 5, st.session_state.num_wrong_answers)

    # Generate questions button
    if st.button("Generate Quiz", key='toggle_option_button', on_click=lambda: toggle_state('generation'), use_container_width=True):
        if prompt_context != '':
            st.session_state.prompt = prompt_context
        else:
            st.error("Please enter some text to generate questions")


# Display Quiz state
elif st.session_state.current_state == 'generation':
    st.empty()
    try:
        prompt_context = st.session_state.prompt
        num_wrong_answers = st.session_state.num_wrong_answers

        # generate questions
        gp_bar = st.progress(0, text="Initializing Quiz...")

        questions = generate_questions(prompt_context, gp_bar)

        questions_array = generate_answers(
            prompt_context, questions, gp_bar)

        st.session_state.quiz = generate_wrong_answers(
            questions_array, num_wrong_answers, gp_bar)

        gp_bar.progress(100, text="Quiz finalized")
        time.sleep(0.5)
        gp_bar.empty()

        st.toast('Your Quiz was successfully generated!', icon='😍')
        toggle_state('display')
    except Exception as e:
        # st.write(str(e))
        if str(e) == "No questions generated":
            st.toast(
                "Whoops, the App didn't find any Question..   . Try again please", icon='😓')
            toggle_state('input')
        else:
            st.toast(
                "There was an error generating your quiz. Try again please, maybe with a longer text", icon='😓')
            toggle_state('input')


# Display quiz questions
if st.session_state.current_state == 'display':
    for question in st.session_state.quiz:
        q, a, c = question.get_answer_outlet_parts()
        stb.single_choice(q, a, c)

    st.markdown('<span style="display: flex;height: 50px;"></span>',
                unsafe_allow_html=True)

    # Button to generate another quiz and hide questions
    st.write('Want to take another quiz?')
    if st.button("Generate Another Quiz", key='toggle_option_button', on_click=lambda: toggle_state('input'), use_container_width=True):
        # reset quiz questions
        st.session_state.quiz = []
