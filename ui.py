import streamlit as st
from random import randint
from streamlit import session_state as state
from io import StringIO
from drawing import predict
import os
import torch
import clip
# from inotify_simple import INotify, flags
import json
# from atomicwrites import atomic_write
from time import sleep
from os.path import exists
from utils import atomic_write

# st.set_page_config(layout="wide")

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        </style> """
st.markdown(hide_menu_style, unsafe_allow_html=True)

header = st.empty()
subheader = st.empty() 
myform = st.empty()
summarizations_form = st.empty()
chart_checkbox = st.empty()

# @st.cache
# def load_clip():
#     state.model, _ = clip.load('ViT-B/32', torch.device('cuda'), jit=False)

if 'checkbox_val' not in st.session_state:
    state.checkbox_val = False
    state.uploaded_file = str(randint(0,10000))
    state.lines = 425
    state.iters = 400
    state.prompt = ""
    state.main_screen = True
    state.summarizations_screen = False
    state.running_screen = False
    state.radiobox = ""
    
if 'model' not in st.session_state:
    with st.spinner('Loading CLIP...'):
      state.model, _ = clip.load('ViT-B/32', torch.device('cuda'), jit=False)

def checkbox_click():
    state.checkbox_val = not state.checkbox_val

def main_screen_submit(lines, iters, prompt, summarizations=False):
    chart_checkbox.empty()
    header.empty()
    subheader.empty()
    myform.empty()
    state.lines = lines
    state.iters = iters
    state.prompt = prompt
    state.main_screen = False
    if summarizations:
        state.summarizations_screen = True
    else:
        state.running_screen = True
    main()

def summarizations_screen_submit():
    header.empty()
    state.main_screen = False
    state.summarizations_screen = False
    state.running_screen = True
    state.prompt = state.radiobox
    main()

def restart():
    header.empty()
    state.prompt = ""
    state.main_screen = True
    main()

def main():
    if state.main_screen:
        state.summarizations_screen = False
        state.running_screen = False
    elif state.summarizations_screen:
        state.running_screen = False

if state.main_screen:
    header.header("Text-to-paiting/drawing generation")
    subheader.write("OpenAI's CLIP + Differentiable Drawing and Sketching")
    chart_checkbox.checkbox('Insert book paragraph or chapter', value=state.checkbox_val, on_change=checkbox_click)
    if state.checkbox_val:
        with myform.form(key='form-charts'):
            lines = st.slider("Number of lines", value = state.lines, min_value = 100, max_value = 850, step = 10)
            iters = st.slider("Number of iterations", value = state.iters, min_value = 100, max_value = 1000, step = 10)
            prompt = st.text_area("Paste the chapter contents here...", height=200)
            uploaded_file = st.file_uploader("or upload .txt file", type="txt", key=st.session_state.uploaded_file)
            submit = st.form_submit_button("Submit")
        if submit:
            string = ""
            if not prompt and uploaded_file is None:
                st.error('Please insert a chapter content.')
                st.stop()
            elif uploaded_file is not None:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                string_data = stringio.read()
                state.uploaded_file = str(randint(0,10000))
                string = string_data
            else:
                string = prompt
            main_screen_submit(lines, iters, string, summarizations=True)
    else:
        with myform.form(key='form-custom'):
            lines = st.slider("Number of lines", value = state.lines, min_value = 100, max_value = 850, step = 10)
            iters = st.slider("Number of iterations", value = state.iters, min_value = 100, max_value = 1000, step = 10)
            prompt = st.text_input("Insert a text prompt for the drawing here", max_chars=100)
            submit = st.form_submit_button("Submit")
        if submit:
            if not prompt:
                st.error('Please insert a prompt.')
                st.stop()
            main_screen_submit(lines, iters, prompt)

if state.summarizations_screen:
    header.subheader('LED-booksum + [T5-headline, YAKE, RAKE, KeyBERT]')
    with st.spinner('Generating the prompts from the text...'):
        atomic_write(state.prompt,'drive/MyDrive/3rd_yr_project/text_to_summarise.txt')
        while os.stat("drive/MyDrive/3rd_yr_project/prompts.json").st_size == 0:
          sleep(2)
        sleep(2)
        with open('drive/MyDrive/3rd_yr_project/prompts.json', 'r+') as f:
            data = json.load(f)
            f.truncate(0)
        if (len(data) != 4): print('error')
        headline = data['headline']
        yake_kw = data['yake']
        rake_kw = data['rake']
        bert_kw = data['bert']
    with summarizations_form.form(key='summarizations_form'):
        prompt = st.radio(
            "Choose one of the generated prompts",
            (headline, yake_kw, rake_kw, bert_kw), key='radiobox')
        st.form_submit_button("Submit", on_click=summarizations_screen_submit)
    st.button('Back', on_click=restart)

if state.running_screen:
    summarizations_form.empty()
    header = st.header(state.prompt) 
    clip_sim = predict(state.model, prompt=state.prompt, update={}, lines=state.lines, iters=state.iters)
    st.success('Done, the clip similarity for the best image was: ' + str(clip_sim))
    st.button("Try again", on_click=restart)
    st.stop()

if __name__ == "__main__":
    main()