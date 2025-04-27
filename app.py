import streamlit as st
import graph
from graph import app
from graph import img
from graph import llm
from graph import LawSection
import asyncio

st.set_page_config(
    page_title="LexMerger",
    page_icon="🚀",
    layout="wide",
)
st.logo("logo.png")

nachweisg = ""
amendments = ""
try:
    # Laden der Änderungen aus dem Bundesgesetzblatt
    with open("changes.txt") as file:
        amendments = file.read()

    # Laden der alten Version des Gesetzes
    with open("nachweisg.txt") as file:
        nachweisg = file.read()
except e:
    print("couldnt read file")

st.title("LexMerger")
st.write(f"{llm.model_name}")

col1, col2, col3 = st.columns(3)

col2.write("### Original")
original_placeholder = col2.empty()

info = col1.empty()
instructlist = col1.empty()
#col1.image(img, caption="Graph", use_container_width=True)

col3.write("### Updated")
status_placeholder = col3.empty()

inputs = {
    "text": nachweisg,
    "amendments": amendments,
    "original": nachweisg,
    "instructions": [],
    "sections": []
}

initial_sections = []
initial_instructions = []

config = {"recursion_limit": 100}
for event in app.stream(inputs, config=config):
    first_key = list(event.keys())[0]
    info.write(first_key)
    if first_key == "init":
        info.write("Verarbeite Änderungsgesetz")
    else:
        info.write(f"{len(event.get(first_key).get("instructions"))} Änderungen gefunden.")
        instructions = event.get(first_key).get("instructions")
        if len(initial_instructions) == 0:
            initial_instructions = instructions
        with instructlist.container():
            for idx, item in enumerate(initial_instructions):
                with st.expander(f"{item.operation} {item.paragraph} {item.selector}"):
                    st.write(f"{item.newvalue}")
                    if item.oldvalue:
                        st.write(f"*Ersetze* {item.oldvalue}")
        sections = event.get(first_key).get("sections", [])
        if len(sections) > 0:
            if len(initial_sections) == 0:
                initial_sections = sections
                with original_placeholder.container():
                    for section in initial_sections:
                        st.write(f"### {section.identifier} {section.title}")
                        st.write(f"{section.content}")
            with status_placeholder.container():
                for section in sections:
                    st.write(f"### {section.identifier} {section.title}")
                    st.write(f"{section.content}")