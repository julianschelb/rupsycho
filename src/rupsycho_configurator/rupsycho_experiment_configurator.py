"""
This module defines a streamlit web-app that allows users to easily create or edit R.U.Psycho experiment 
configurations. The app is intended to be run locally by the user.

The app currently only supports the import but not the configuration of the following experiment sections:
- 'parameters'
- 'prompt_template'
- 'models'
- 'attributes'
The concept of 'default_answer_options' is not supported at all.
"""

import os
import streamlit as st
import pandas as pd
import json
from copy import deepcopy
import traceback
from langchain_openai import ChatOpenAI
import openai
import time
import utils
from questionnaire import Questionnaire





# Setup of session state
# ----------------------
# All experiment elements (i.e. profiles, questionnaire items, answer options, etc.) are stored in session state
# as dictionaries and have corresponding input widgets that can update their values.
#
# There are other session_state variables that are implicitly created by the definition of the input widget 
# that is used to change their value. They are retrievable from session state via the key of their widget.
#
# The usual flow of information is: 
#   1. user creates new input widget for experiment element and puts in data
#   2. input widget passes this data on to its corresponding element in session state 
#   3. all elements from session state are written to output
# For special cases where a non-empty experiment element is created internally, its data is indirectly 
# passed to the new corresponding widgets.

# Definitions of the formats that represent experiment elements.
# When adding new elements to the session state, a copy of the corresponding template is made, edited and added.
# Adding new keys in these templates will therefore be effective globally.
profile_template = {
    "title": "", 
    "name": "", 
    "id": None, # position of the profile in the list; is updated on each rerun
    "ethnicity": "",
    "widget_key": None
} 
item_template = {
    "question": "", 
    "reversed": False,
    "answer_options": {}, 
    "attributes": {},
    "widget_key": None,
    "next_answer_widget_key": 0,
}
answer_template = {
    "text": "", 
    "weight": None, 
    "ignored_for_scale": False,
    "widget_key": None,
}

# Widget keys uniquely identify input widgets. Each new experiment element gets one for its widgets.
# next_ variables are not used on their own as widget keys but as part of informative strings. 
# Increment next_ variables after each use!
if 'next_profile_widget_key' not in st.session_state:
    st.session_state.next_profile_widget_key = 0

if 'next_item_widget_key' not in st.session_state:
    st.session_state.next_item_widget_key = 0

if 'next_global_ans_widget_key' not in st.session_state:
    st.session_state.next_global_ans_widget_key = 0

if 'quest_text' not in st.session_state:
    st.session_state.quest_text = '' # placeholder; is later overwritten by widget

# error that will be rendered on the next rerun of the script
if 'error_message' not in st.session_state:
    st.session_state.error_message = ''

if 'valid_api_key' not in st.session_state:
    st.session_state.valid_api_key = False

if 'page_range' not in st.session_state:
    st.session_state.page_range = (0, 0)

if 'quest_pages' not in st.session_state:
    st.session_state.quest_pages = []

if 'exp_name' not in st.session_state:
    st.session_state.exp_name = ''

if 'exp_descr' not in st.session_state:
    st.session_state.exp_descr = ''

if 'quest_name' not in st.session_state:
    st.session_state.quest_name = ''

if 'quest_instr' not in st.session_state:
    st.session_state.quest_instr = ''

if 'parameters' not in st.session_state:
    st.session_state.parameters = {}

if 'prompt_template' not in st.session_state:
    st.session_state.prompt_template = {
        "type": "chat",
        "messages": [
            {
                "role": "system",
                "content": "Objective: Act like you are {persona_description}, a survey participant answering a questionnaire.\n{general_instruction}\nInstructions: Choose from the list of answer options to answer the question. Answer the question using only the provided answer options. If none of the options are correct, choose the option that is closest to being correct. The solution must be provided in this format: {{\"answer\": \"answer option\"}}"
            },
            {
                "role": "user",
                "content": "Question: {question}\nAnswer Options: {answer_options}\nAnswer:"
            }
        ]
    }

if 'models' not in st.session_state:
    st.session_state.models = {}

if 'attributes' not in st.session_state:
    st.session_state.attributes = {}

# stores all current demographic profile elements
if 'dem_profiles' not in st.session_state: 
    default_profile = deepcopy(profile_template)
    default_profile.update({'id': 1, 'widget_key':st.session_state.next_profile_widget_key})
    st.session_state.next_profile_widget_key = st.session_state.next_profile_widget_key + 1
    st.session_state.dem_profiles = [default_profile]

def add_answer(idx: int):
    """Add a new empty answer option to the questionnaire item at index 'idx'."""
    item = st.session_state.quest_items[idx]
    length = len(item['answer_options'])
    ans = deepcopy(answer_template)
    ans.update({
        "weight": length, 
        "widget_key": item['next_answer_widget_key'],
    })
    item['answer_options'].update({f"{length+1}": ans})
    item['next_answer_widget_key'] = item['next_answer_widget_key'] + 1

# stores all current questionnaire item elements
if 'quest_items' not in st.session_state:
    default_item = deepcopy(item_template)
    default_item.update({"widget_key": st.session_state.next_item_widget_key})
    st.session_state.next_item_widget_key = st.session_state.next_item_widget_key + 1
    st.session_state.quest_items = [default_item]
    # add two empty answer options by default
    add_answer(0)
    add_answer(0)

def add_global_answer():
    """Add a new empty global answer option to the session state."""
    length = len(st.session_state.global_answer_set)
    ans = deepcopy(answer_template)
    ans.update({
        "weight": length, 
        "widget_key": st.session_state.next_global_ans_widget_key,
    })
    st.session_state.global_answer_set.append(ans)
    st.session_state.next_global_ans_widget_key = st.session_state.next_global_ans_widget_key + 1

# stores all current global answer option elements
if 'global_answer_set' not in st.session_state:
    st.session_state.global_answer_set = []
    # add two empty answer options by default
    add_global_answer()
    add_global_answer()



# Streamlit page
# --------------
# Each rerun of the script iterates over all experiment elements and renews the rendering of their input widgets.
# For this it is crucial that the identifying information of a widget (label, min or max value, 
# default value, placeholder text, help text, and key) stays unchanged across reruns, otherwise streamlit will not
# recognize the widget and creates a new one instead .

st.set_page_config(layout="wide")

# smaller margin at the top of the page
st.markdown("""
    <style>
        .block-container {
            padding-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

col0_1, col0_2 = st.columns([0.7, 0.3], gap="small", vertical_alignment='bottom')
with col0_1:
    st.title("R.U.Psycho Experiment Configurator")
with col0_2:
    notification_container = st.container()
    if st.session_state.error_message:
        with notification_container:
            st.error(st.session_state.error_message, icon=":material/warning:")

col1, col2 = st.columns(2, gap="medium")
with col1:
    st.header(body='Configurator', help=None)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Tools**", "**Experiment Info**", "**Demographic Profiles**", "**Questionnaire Info**", "**Questionnaire Items**"])
    
    with tab1:
        with st.expander(
            label="Language Model",
            icon=":material/auto_fix_high:",
        ):
            instructions = "Let a language model (OpenAI GPT-4o mini) automatically create the questionnaire-section of the configuration for you. Put in an OpenAI API-key below and provide your questionnaire either by uploading it as a PDF or by directly pasting its text into the input field. Note that this will overwrite the current questionnaire-section of the configuration."
            st.caption(instructions)
            def validate_api_key():
                """Validate the API key by sending a trivial API call and checking if it is answered as expected."""
                with notification_container:
                    client = openai.OpenAI(api_key=st.session_state.api_key)
                    try:
                        _ = client.models.list()
                    except openai.AuthenticationError:
                        # print(traceback.format_exc())
                        st.error('Invalid API key', icon=":material/warning:")
                        st.session_state.valid_api_key = False
                        st.session_state.api_key = ''
                    except Exception:
                        print(traceback.format_exc())
                        st.error('Something went wrong', icon=":material/warning:")
                        st.session_state.valid_api_key = False
                        st.session_state.api_key = ''
                    else:
                        st.success('Valid API key', icon=":material/done:")
                        st.session_state.valid_api_key = True
                        os.environ["OPENAI_API_KEY"] = st.session_state.api_key

            st.text_input(
                label="OpenAi API key",
                label_visibility='collapsed',
                placeholder="OpenAi API key",
                key="api_key",
                type='password',
                on_change=validate_api_key,
            )

            def concat_pages(a: int, b: int) -> str:
                """Return the string of the concatenated pages 'a' up to but not including 'b'."""
                return '\n\n'.join(st.session_state.quest_pages[a:b])
            
            def set_or_delete_pdf():
                """Update session state to reflect the change the user made on the value of the PDF input widget."""
                if st.session_state.uploaded_pdf is not None:
                    st.session_state.quest_pages = utils.extract_quest_pages(st.session_state.uploaded_pdf)
                    st.session_state.page_range = (1, len(st.session_state.quest_pages))
                    st.session_state.quest_text = concat_pages(0, st.session_state.page_range[1])
                else:
                    # if the user removes the questionnaire pdf -> leave the displayed text untouched but remove pdf content and slider in background
                    st.session_state.quest_pages = []
                    st.session_state.page_range = (0, 0)
            
            st.file_uploader(
                label="Upload and load a questionnaire",
                label_visibility='collapsed',
                key='uploaded_pdf',
                type='pdf',
                accept_multiple_files=False,
                on_change=set_or_delete_pdf,
            )

            def set_text_range():
                """Set the displayed questionnaire text to the concatenated text of the pages that the currently set page range covers."""
                st.session_state.quest_text = concat_pages(st.session_state.page_range[0]-1, st.session_state.page_range[1])
            
            if len(st.session_state.quest_pages) > 1:
                st.slider(
                    label="Select pages",
                    key='page_range',
                    min_value=1,
                    max_value=len(st.session_state.quest_pages),
                    on_change=set_text_range,
                )

            def reset_app():
                """Delete all directly output-relevant elements from session state.
                
                The app needs to be rerun after calling this function because the here deleted values
                need to be set up for the app to function. This is implicitly given if used for the 
                error handling of a callback function of a widget.
                """
                st.session_state.pop(key='exp_name', default=None)
                st.session_state.pop(key='exp_descr', default=None)
                st.session_state.pop(key='quest_name', default=None)
                st.session_state.pop(key='quest_instr', default=None)

                st.session_state.pop(key='parameters', default=None)
                st.session_state.pop(key='prompt_template', default=None)
                st.session_state.pop(key='models', default=None)
                st.session_state.pop(key='attributes', default=None)

                st.session_state.pop(key='dem_profiles', default=None)
                st.session_state.pop(key='quest_items', default=None)
                st.session_state.pop(key='global_answer_set', default=None)

            def reset_and_rerun_app():
                """Delete all directly output-relevant elements from session state and rerun the script from the top.
                
                Be very carful to only use this function if it is absolutely sure that it will not be called again during the rerun. 
                Otherwise it will result in an endless loop of reruns!
                """
                reset_app()
                st.rerun()

            def run_model():
                """Run the model with a single prompt that contains the currently displayed questionnaire text and add the output as new experiment elements."""
                st.session_state.step = "step2"
                with notification_container:
                    try:
                        
                        st.session_state.use_global_answer_set = False
                        init_new_answer_mode()
                        st.session_state.quest_items = []
                        
                        model = ChatOpenAI(
                            model="gpt-4o-mini",
                            api_key=os.getenv('OPENAI_API_KEY'),
                            temperature=0.8,
                            max_tokens=16384, # 2048 4096 8192 16384
                        )
                        # removes special characters so the model doesn't have to deal with escaping them
                        cleaned_text = utils.rmv_special_chars(st.session_state.quest_text) 

                        # use zero shot prompt because it slightly outperformed one shot prompt in experiments
                        prompt = utils.mk_prompt(cleaned_text)
                        print(f"+++++ Prompt +++++\n{prompt.to_string()}\n++++++++++++++++++")

                        chain = model | utils.extract_json
                        name, instr, questions, answers = chain.invoke(prompt)
                        quest = Questionnaire('', '', questions, answers)
                        questions_answers = quest.get_merged_questions_answers()

                        # print extracted model outputs
                        print("Model output:\n------------\n")
                        print(f"Name:\n---------\n{name}\n")
                        print(f"Instructions:\n---------\n{instr}\n")
                        utils.list_pretty_print(questions, 'questions')
                        utils.list_pretty_print(answers, 'answer sets')

                        st.session_state.quest_name = name
                        st.session_state.quest_instr = instr

                        # adds questions and answer sets to session state
                        for q, anss in questions_answers:
                            answer_options = {}
                            for i, ans in enumerate(anss):
                                ans_dict = deepcopy(answer_template)
                                ans_dict.update({
                                    "text": ans,
                                    "weight": i,
                                })
                                answer_options.update({f"{i}": ans_dict })
                            add_predef_item(question=q, reversed=False, predef_answers=answer_options)

                        update_item_widgets_values()
                        st.success('Model run was successful', icon=":material/done:")

                    except Exception:
                        print(traceback.format_exc())
                        st.error('Model error, please try again', icon=":material/warning:")
                        reset_app()

            st.button(
                label="**Run**",
                on_click=run_model,
                use_container_width=True,
                icon=":material/auto_fix_high:",
                disabled=not (st.session_state.quest_text and st.session_state.valid_api_key)
            )

        def set_profile_widgets_values():
            """Set the values of the input widgets of all profile elements to their corresponding value in session state. 
            
            Call this method after adding new non-empty profiles and before the next rerun. This pre-sets the values of 
            the corresponding new input widgets so that they directly display the predefined text when they are defined 
            during the next rerun.
            The values are set indirectly via a workaround. Directly setting the default values would cause troubles 
            with streamlit as it does not intend for widget default values to be set directly via session state 
            in the definition of the widget.
            """
            for p in st.session_state.dem_profiles:
                # uses the same widget key that the definition of the widget will use to preset a default value of the widget
                key = f"title_{p['widget_key']}"
                st.session_state[key] = p['title']
                key = f"name_{p['widget_key']}"
                st.session_state[key] = p['name']
                key = f"ethnicity_{p['widget_key']}"
                st.session_state[key] = p['ethnicity']


        def update_item_widgets_values():
            """Set the values of the input widgets of all questionnaire item elements to their corresponding value in session state. 
            
            Call this method after adding new non-empty items and before the next rerun. This pre-sets the values of 
            the corresponding new input widgets so that they directly display the predefined text when they are defined 
            during the next rerun.
            The values are set indirectly via a workaround. Directly setting the default values would cause troubles 
            with streamlit as it does not intend for widget default values to be set directly via session state 
            in the definition of the widget.
            """
            for item in st.session_state.quest_items:
                key = f"question_{item['widget_key']}"
                st.session_state[key] = item['question']

                for ans in item['answer_options'].values():
                    key = f"ans_{ans['widget_key']}_item_{item['widget_key']}"
                    st.session_state[key] = ans['text']
        
        def add_predef_profile(title: str, name: str, ethnicity: str):
            """Add a new non-empty profile to session state with the given values."""
            prof = deepcopy(profile_template)
            prof.update({
                "title": title, 
                "name": name, 
                "ethnicity": ethnicity,
                "widget_key": st.session_state.next_profile_widget_key
            })
            st.session_state.dem_profiles.append(prof)
            st.session_state.next_profile_widget_key = st.session_state.next_profile_widget_key + 1

        def add_predef_item(question: str, reversed: bool, predef_answers: dict[any], attributes: dict[any] = None):
            """Add a new non-empty questionnaire item to session state with the given values."""
            # avoids 'mutable default parameter pitfall'
            if attributes is None:
                attributes = {}
            # adds item to session state
            item = deepcopy(profile_template)
            item.update({
                "question": question, 
                "reversed": reversed,
                "answer_options": {},
                "attributes": attributes,
                "widget_key": st.session_state.next_item_widget_key,
                "next_answer_widget_key": len(predef_answers),
            })
            st.session_state.quest_items.append(item)
            st.session_state.next_item_widget_key = st.session_state.next_item_widget_key + 1

            # adds answer options to item
            for i, predef_ans in enumerate(predef_answers.values()):
                
                ans = deepcopy(answer_template)
                ans.update({
                    "text": predef_ans['text'],
                    "weight": predef_ans['weight'],
                    "ignored_for_scale": predef_ans['ignored_for_scale'],
                    "widget_key": i,
                })
                item['answer_options'].update({
                    f"{i+1}": ans
                })

        with st.expander(
            label="Import Configuration",
            icon=":material/file_upload:",
        ):
            def import_config():
                """Replace the entire current config with the config that was uploaded by the user.

                The new config is accepted if ALL expected keys ('name', 'attributes', 'parameters', items, answers etc.) 
                are found in the correct positions. If the setup is rejected at any point during 
                importing, the entire config so far is deleted. There are no explicit type checks of values. 
                So it is possible that configs with invalid structure are imported. This can lead to situations where unintended 
                values pass into session state that cause exceptions during widget rendering. This is handled by a reset of 
                the app. E.g. if the value of the experiment name is a dict instead of a string.
                """
                if st.session_state.uploaded_config is not None:
                    st.session_state.use_global_answer_set = False
                    init_new_answer_mode()
                    st.session_state.quest_items = []
                    st.session_state.dem_profiles = []

                    with notification_container:
                        try:
                            config = json.load(st.session_state.uploaded_config)

                            st.session_state.exp_name = config['name']
                            st.session_state.exp_descr = config['description']
                            st.session_state.parameters = config['parameters']
                            st.session_state.prompt_template = config['prompt_template']
                            st.session_state.models = config['models']
                            
                            # profiles
                            for p in config['demographic_profiles'].values():
                                p = p['attributes']
                                add_predef_profile(p['title'], p['name'], p['ethnicity'])
                            set_profile_widgets_values()

                            # questionnaire
                            quest = config['questionnaire']
                            st.session_state.quest_name = quest['name']
                            st.session_state.quest_instr = quest['general_instruction']
                            st.session_state.attributes = quest['attributes']
                            for item in quest['instruction_items']:
                                add_predef_item(item['question'], item['reversed'], item['answer_options'], item['attributes'])
                            update_item_widgets_values()
                            st.success('Configuration loaded', icon=":material/done:")
                        
                        except Exception:
                            print(traceback.format_exc())
                            st.error('Invalid configuration', icon=":material/warning:")
                            print('reset this bs')
                            reset_app()
                else:
                    pass # if imported config file is removed by the user -> do nothing
                        
            instructions = "Upload and load an existing configuration for editing. Note that this will overwrite the current configuration."
            st.caption(instructions)
            st.file_uploader(
                label="Upload and load an existing config",
                label_visibility='collapsed',
                key='uploaded_config',
                type='json',
                accept_multiple_files=False,
                # label_visibility='hidden',
                on_change=import_config,
            )

        with st.expander(
            label='Help',
            icon=":material/help:"
        ):
            paper_url = "https://doi.org/10.48550/arXiv.2503.10229"
            help = f"This configurator facilitates the creation of new and editing of existing experiment configurations for the R.U.Psycho framework.\n\nA guide on how to use the configurator can be found in the section 'Using the Configurator App' in the README.md file of the package.\n\nInformation about the structure of the configuration format and its components can be found in the section '3.1 Experiment Definition' of the [paper]({paper_url})."
            st.markdown(help)

    with tab2:
        try:
            st.text_input(
                label="Experiment name",
                key='exp_name',
                disabled=False,
            )
            st.text_area(
                label="Experiment description",
                key='exp_descr',
                height=150,
            )
        except Exception:
            print(traceback.format_exc())
            st.session_state.error_message = "Unexpected value in 'Experiment Info', app was reset"
            reset_and_rerun_app()
            
        
    with tab3:
        def add_empty_profile():
            """Add a new empty profile element to the session state whose widgets will be rendered during the next rerun."""
            prof = deepcopy(profile_template)
            prof.update({"widget_key": st.session_state.next_profile_widget_key})
            st.session_state.dem_profiles.append(prof)
            st.session_state.next_profile_widget_key = st.session_state.next_profile_widget_key + 1
            
        def delete_profile(prof_idx: int):
            """Delete the profile element at the specified index and its corresponding input widgets."""
            # not rendering a preexisting widget on a rerun automatically deletes the widget object
            st.session_state.dem_profiles.pop(prof_idx)
            # st.session_state.profile_key_ids.pop(idx)

        def duplicate_profile(prof_idx: int):
            """Copy the profile element at the given index and insert the copy as a unique new item into the session state at the position after the template profile."""
            templ = st.session_state.dem_profiles[prof_idx-1]
            dupl = deepcopy(templ)
            dupl.update({"widget_key": st.session_state.next_profile_widget_key,})
            st.session_state.dem_profiles.insert(prof_idx, dupl)
            st.session_state.next_profile_widget_key = st.session_state.next_profile_widget_key + 1

            # indirectly sets value of new input widgets
            key=f"title_{dupl['widget_key']}"
            st.session_state[key] = dupl['title']
            key=f"name_{dupl['widget_key']}"
            st.session_state[key] = dupl['name']
            key=f"ethnicity_{dupl['widget_key']}"
            st.session_state[key] = dupl['ethnicity']

        def import_profiles():
            """Replace the current profile elements in session state with the profiles specified in the CSV file uploaded by the user.
            
            The file is accepted if its header contains 'title', 'name' and 'ethnicity' and no value is empty.
            If the file is rejected at any point, all profile elements that were added so far are deleted.
            There are no explicit type checks of values. This can lead to situations where unintended values 
            pass into session state that cause exceptions during widget rendering. This is handled by a reset 
            of the app.
            """
            if st.session_state.uploaded_csv:
                with notification_container:
                    try:
                        st.session_state.dem_profiles = []
                        profiles = pd.read_csv(st.session_state.uploaded_csv)
                        
                        if not all(attr in profiles.columns.to_list() for attr in ['title', 'name', 'ethnicity']):
                            # missing attribute
                            raise TypeError
                        if profiles.isna().any().any():
                            # NaN value
                            raise TypeError
                        
                        for _, p in profiles.iterrows():
                            add_predef_profile(p['title'], p['name'], p['ethnicity'])

                        set_profile_widgets_values()

                    except Exception:
                        print(traceback.format_exc())
                        st.session_state.pop(key='dem_profiles', default=None)
                        st.error('Invalid file structure', icon=":material/warning:")
            else:
                pass # if user removes file do nothing

        with st.expander(label="Import from CSV", icon=":material/file_upload:",):
            instructions = "Load a list of demographic profiles that are specified in a CSV file into the experiment configuration. Note that this will overwrite the current profiles. The header of the file has to contain 'title', 'name' and 'ethnicity' and no value may be left out."
            st.caption(instructions)
            st.file_uploader(
                label="Import from CSV",
                key='uploaded_csv',
                type='csv',
                on_change=import_profiles,
                label_visibility='collapsed',
            )
        
        st.button(
            label="Add profile",
            on_click=add_empty_profile,
            icon=":material/add_box:", # add_box add add_circle_outline
            use_container_width=False,
        ) 
        
        # profiles widgets
        for i, profile in enumerate(st.session_state.dem_profiles):
            
            profile['id'] = i+1
            # text widgets update the list of profiles in session state on change
            with st.container(border=True):
                col1_1, col1_2, col1_3 = st.columns([0.8, 0.1, 0.1], gap="small", vertical_alignment='bottom')
                with col1_1:
                    st.subheader(f"Profile {i+1}")
                
                with col1_2:
                    st.button(
                        label='',
                        key=f"duplicate_profile_button_{profile['widget_key']}",
                        args=(i+1,), 
                        on_click=duplicate_profile,
                        use_container_width=True,
                        help='Duplicate profile',
                        icon=":material/library_add:",
                    )
                with col1_3:
                    st.button(
                        label="",
                        key=f"delete_profile_button_{profile['widget_key']}", 
                        args=(i, ), 
                        help="Delete profile",
                        on_click=delete_profile, 
                        use_container_width=True, 
                        icon=":material/delete_outline:", # delete_outline close
                    )

                # renders profile input widgets
                try:
                    profile['title'] = st.text_input(
                        label='title', 
                        key=f"title_{profile['widget_key']}", 
                        placeholder="Title", 
                        label_visibility='collapsed'
                    )
                    profile['name'] = st.text_input(
                        label='name', 
                        label_visibility='collapsed',
                        key=f"name_{profile['widget_key']}", 
                        placeholder="Name", 
                    )
                    profile['ethnicity'] = st.text_input(
                        label='ethnicity',
                        label_visibility='collapsed',
                        key=f"ethnicity_{profile['widget_key']}",
                        placeholder="Ethnicity", 
                    )
                except Exception:
                    print(traceback.format_exc())
                    st.session_state.error_message = "Unexpected value in 'Demographic Profiles', app was reset"
                    reset_and_rerun_app()

            st.write('')
    
    with tab4:
        try:
            st.text_input(
                label="Questionnaire name",
                key='quest_name',
            )
            st.text_area(
                label="Questionnaire instructions",
                key='quest_instr',
                height=150,
            )
        except Exception:
            print(traceback.format_exc())
            st.session_state.error_message = "Unexpected value in 'Questionnaire Info', app was reset"
            reset_and_rerun_app()

    with tab5:
        def add_item():
            """Add a new empty questionnaire item to the session state whose input widgets will be rendered during the next rerun."""
            item = deepcopy(item_template)
            item.update({"widget_key": st.session_state.next_item_widget_key,})
            st.session_state.quest_items.append(item)
            st.session_state.next_item_widget_key = st.session_state.next_item_widget_key + 1

            # add 2 empty answer options by default
            item_idx = len(st.session_state.quest_items)-1
            add_answer(item_idx)
            add_answer(item_idx)

        def delete_all_local_answer_sets():
            """Set the 'answer_options' attr of every item to the empty dict (not deleting it!)."""
            for item in st.session_state.quest_items:
                item['answer_options'] = {}

        col1_1, col1_2 = st.columns([0.3, 0.7], gap="small", vertical_alignment='bottom')
        with col1_1:
            st.button(
                label="Add item",
                key='add_item_button',
                on_click=add_item,
                use_container_width=False,
                icon=":material/add_box:",
            )
        with col1_2:
            def init_new_answer_mode():
                """Set up the app for the use of either the local or global answer set depending on what the toggle is set to."""
                delete_all_local_answer_sets()
                if not st.session_state.use_global_answer_set:
                    st.session_state.pop('global_answer_set')
                    for idx in range(len(st.session_state.quest_items)):
                        add_answer(idx)
                        add_answer(idx)
                    
            st.toggle(
                label="Global answer set",
                key='use_global_answer_set',
                help="Define a single answer set that will be used for all questionnaire items. Note that this will overwrite the current answer sets.",
                on_change=init_new_answer_mode,
            )

        # global answer options
        if st.session_state.use_global_answer_set:
            with st.expander(
                label="**Global answer options set**",
                expanded=True,
            ):  
                _, col1_1, col1_2 = st.columns([0.05, 0.5, 0.45], gap="small", vertical_alignment='center')
                with col1_1:
                    st.button(
                        label="Add global answer option",
                        key="add_global_ans_button",
                        on_click=add_global_answer,
                        use_container_width=False,
                        icon=":material/add_box:",
                    )
                with col1_2:
                    st.toggle(
                        label="Reversed scoring",
                        key=f"global_reversed",
                    )

                # renders global answer input widgets
                for j, ans_opt in enumerate(st.session_state.global_answer_set):
                    # updates weights of answer options in case an answer was deleted
                    ans_opt.update({"weight": j})

                    col1_1, col1_2, col1_3 = st.columns([0.05, 0.85, 0.1], gap="small", vertical_alignment='center')
                    with col1_1:
                        st.markdown(f"**{j+1}**")
                    with col1_2:
                        ans_opt['text'] = st.text_input(
                            label=f"Answer option",
                            label_visibility='collapsed',
                            key=f"global_ans_{ans_opt['widget_key']}",
                        )
                    
                    def delete_global_answer(ans_idx: int):
                        """Delete the global answer at the specified index."""
                        del st.session_state.global_answer_set[ans_idx]
                        
                    with col1_3:
                        st.button(
                            label="",
                            key=f"del_global_ans_{j}",
                            args=(j,),
                            on_click=delete_global_answer,
                            icon=":material/close:", # delete_outline close
                            use_container_width=True
                        )
                
                # sets the answer options of all items to the current global answer set
                delete_all_local_answer_sets()
                for item in st.session_state.quest_items:
                    item['reversed'] = st.session_state.global_reversed
                    for j, ans_opt in enumerate(st.session_state.global_answer_set):
                        item['answer_options'].update({f"{j+1}": ans_opt})


        
        def delete_item(item_idx: int):
            """Delete the questionnaire item at the specified location and its input widgets from session state."""
            # input widgets are implicitly deleted
            tmp = st.session_state.quest_items.pop(item_idx)

        def duplicate_item(idx: int):
            """Copy the questionnaire item at the given index and insert the copy as a unique new item into the session state at the position after the template item."""
            templ = st.session_state.quest_items[idx-1]
            dupl = deepcopy(templ)
            dupl.update({"widget_key": st.session_state.next_item_widget_key,})
            st.session_state.quest_items.insert(idx, dupl)
            st.session_state.next_item_widget_key = st.session_state.next_item_widget_key + 1

            # indirectly sets value of new input widget
            key = f"question_{dupl['widget_key']}"
            st.session_state[key] = dupl['question']

            if not st.session_state.use_global_answer_set:
                for ans in dupl['answer_options'].values():
                    key = f"ans_{ans['widget_key']}_item_{dupl['widget_key']}"
                    st.session_state[key]= ans['text']
        
        # questionnaire items
        for i, item in enumerate(st.session_state.quest_items):
            with st.container(border=True):
                col1_1, col1_2, col1_3 = st.columns([0.8, 0.1, 0.1], gap="small", vertical_alignment='center')
                with col1_1:
                    st.subheader(f"Item {i+1}")
                    # st.write(f"**Item {i+1}**")
                    pass
                with col1_2:
                    st.button(
                        label='',
                        key=f"duplicate_item_button_{item['widget_key']}",
                        args=(i+1,), on_click=duplicate_item,
                        use_container_width=True,
                        help='Duplicate item',
                        icon=":material/library_add:", # delete_outline close
                    )
                with col1_3:
                    st.button(
                        label='', # Delete item
                        key=f"delete_item_button_{item['widget_key']}",
                        args=(i,), 
                        on_click=delete_item,
                        help='Delete item',
                        use_container_width=True,
                        icon=":material/delete_outline:", # delete_outline close
                    )
                try:
                    item['question'] = st.text_input(
                        label='question',
                        key=f"question_{item['widget_key']}",
                        placeholder=f"Question",
                        label_visibility='collapsed',
                    )
                except Exception:
                    print(traceback.format_exc())
                    st.session_state.error_message = "Unexpected value in 'Questionnaire Items', app was reset"
                    reset_and_rerun_app()
                
                # local answer options
                if not st.session_state.use_global_answer_set:
                    # answer options controls
                    with st.expander(label='**Answer options**', expanded=False):
                        _, col1_1, col1_2 = st.columns([0.05, 0.5, 0.45], gap="small", vertical_alignment='center')
                        try:
                            with col1_1:
                                st.button(
                                    label="Add answer option",
                                    key=f"add_ans_button_item{item['widget_key']}",
                                    args=(i,),
                                    on_click=add_answer,
                                    use_container_width=False,
                                    icon=":material/add_box:",
                                )
                            with col1_2:
                                item['reversed'] = st.toggle(
                                    label="Reversed scoring",
                                    key=f"reversed_{item['widget_key']}",
                                    # value=item['reversed'],
                                )
                            
                            # renders input widgets of the answer options 
                            answer_options = item['answer_options']
                            for j, ans_opt_key in enumerate(list(answer_options)):
                                # updates dict key and weight in case an answer was deleted
                                # after the for loop the dict has its original order again
                                new_key = f"{j+1}"
                                answer_options[new_key] = answer_options.pop(ans_opt_key)
                                ans_opt = answer_options[new_key]
                                ans_opt.update({"weight": j})

                                col1_1, col1_2, col1_3 = st.columns([0.05, 0.85, 0.1], gap="small", vertical_alignment='center')
                                with col1_1:
                                    st.markdown(f"**{j+1}**")
                                with col1_2:
                                    ans_opt['text'] = st.text_input(
                                        label=f"Answer option", #  {j+1}
                                        label_visibility='collapsed',
                                        key=f"ans_{ans_opt['widget_key']}_item_{item['widget_key']}",
                                    )
                                with col1_3:
                                    def del_local_ans(item_idx: int, ans_key: int):
                                        """From item with index 'item_idx' delete the local answer option with the key/index 'ans_key' and its input widget."""
                                        ans_options = st.session_state.quest_items[item_idx]['answer_options']
                                        ans_options.pop(f"{ans_key}")

                                    st.button(
                                        label="",
                                        key=f"del_ans_{ans_opt['widget_key']}item_{item['widget_key']}",
                                        args=(i, j+1),
                                        on_click=del_local_ans,
                                        icon=":material/close:", # delete_outline close
                                        use_container_width=True
                                    )
                        except Exception:
                            print(traceback.format_exc())
                            st.session_state.error_message = "Unexpected value in 'Questionnaire Items', app was reset"
                            reset_and_rerun_app() 
            st.write('')        
              

# prepares the output json
with col2:
    out_json = {
        "name": st.session_state.exp_name,
        "description": st.session_state.exp_descr,
        "parameters": st.session_state.parameters,
        "prompt_template": st.session_state.prompt_template,
        "models": st.session_state.models,
        "demographic_profiles": {},
        "questionnaire": {
            "name": st.session_state.quest_name,
            "general_instruction": st.session_state.quest_instr,
            "attributes": st.session_state.attributes,
            # "default_answer_options": {},
            "instruction_items": []
        }
    }
    # add demographic profiles
    for profile in st.session_state.dem_profiles:
        out_json['demographic_profiles'].update(
            {
                f"{profile['id']}-{profile['title']} {profile['name']}": {
                    "attributes": {
                        "title": profile['title'],
                        "name": profile['name'],
                        "ethnicity": profile['ethnicity'],
                        "id": profile['id'],
                        # "widget_key": profile['widget_key'], 
                    }, 
                    "template": "{title} {name}"
                }
            
            }
        )
    # add questionnaire items
    for item in st.session_state.quest_items:
        anss = item['answer_options']
        out_json['questionnaire']['instruction_items'].append(
            {
                "question": item['question'], 
                "reversed": item['reversed'], 
                "widget_key": item['widget_key'], 
                "answer_options": {key: 
                    {
                        "text": anss[key]['text'], 
                        "weight": anss[key]['weight'], 
                        "ignored_for_scale": anss[key]['ignored_for_scale'],
                        # "widget_key": anss[key]['widget_key'],
                    } for key in anss}, 
                "attributes": item['attributes'],   
            }
        )
    # automatically escapes any input characters that could disrupt the json format
    json_string = json.dumps(out_json, indent=4)

    col2_1, col2_2 = st.columns([0.75, 0.25], gap="small", vertical_alignment='bottom')
    with col2_1:
        st.header("Input / Output")

        with col2_2:
            st.download_button(
                label="Download",
                data=json_string,
                file_name="rupsycho_experiment_config.json",
                mime="application/json",
                help="Download the current configuration as a .json file",
                icon=":material/file_download:",
                use_container_width=True,
            )
    
    tab1, tab2 = st.tabs([ "**Questionnaire Text**", "**Resulting Configuration**"])
    with tab1:
        st.text_area(
            label="Questionnaire Text",
            key='quest_text',
            height=650,
            placeholder="Cleaning the text by removing irrelevant sections and by correcting formatting mistakes can significantly improve the output quality of the language model.",
            label_visibility='collapsed',
            disabled=False,
        )

    with tab2:
        with st.container(height=650, border=True,):
            # does not display special characters as escaped but returns them as such
            st.json(out_json)

st.session_state.pop('error_message')

print('------- END OF RUN ------- ')