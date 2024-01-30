import streamlit as st


def app_start():
    # st.empty()

    # Stylings
    style = """
        <style>
                .block-container {
                    padding-top: 2rem;
                    padding-bottom: 2rem;
                }
                [data-testid=stSidebarContent] .block-container {
                    padding-top: 0;
                }
                [data-testid=stSidebarContent] .stHeadingContainer h1 {
                    padding-top: 0;
                }
        </style>
        """
    st.markdown(style, unsafe_allow_html=True)

    # START
    st.title("Quiz Generator")
    st.sidebar.title("Quiz Generator Options")

    st.sidebar.info(
        'The quiz generation takes more time the more wrong answers you choose')
