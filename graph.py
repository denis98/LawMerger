

from dotenv import dotenv_values
import getpass
import os

import pickle

from langgraph.graph import END
from langgraph.graph import StateGraph, START
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_groq import ChatGroq
from langchain_community.chat_models.deepinfra import ChatDeepInfra

config = dotenv_values(".env")
os.environ["DEEPINFRA_API_TOKEN"] = config["DEEPINFRA_API_TOKEN"]
os.environ["GROQ_API_KEY"] = config["GROQ_API_KEY"]

from typing import TypedDict, Annotated, List, Union, Literal
from pydantic import BaseModel, Field

# (Gesetz, Paragraph, Selector, Operation, newvalue, oldvalue) 
# Operation: Wähle zwischen "change", "drop", "insert"
class Instruction(BaseModel):
    law: str = Field("Name des betroffenen Gesetzes")
    paragraph: str = Field("Paragraph, in dem die Änderung vorgenommen werden soll. Beispiel: '§ 1'")
    selector: str = Field("Betroffener Abschnitt, z.B. 'Satz 1'")
    operation: Literal["change", "drop", "insert"]
    newvalue: str = Field("Neuer Text (nur befüllen falls zutreffend)")
    oldvalue: str = Field("Alter Text (nur befüllen falls vorhanden und es für die Operation sinn macht)")

    
class Instructions(BaseModel):
    instructions: list[Instruction]


class LawSection(BaseModel):
    law: str = Field("Ausgeschriebener offizieller Name des Gesetzes")
    identifier: str = Field("""Paragraph des Abschnittes. Das Feld muss eindeutig sein. Verwende nur den Paragraphen, erstelle nicht für jeden Absatz einen neuen Eintrag. Beispiele '§ 1', '§ 8'""")
    title: str = Field("Titel des Paragraphen (falls vorhanden)")
    content: str = Field("Inhalt des Abschnittes. Auf keinen Fall verändern oder kürzen.")

class LawText(BaseModel):
    sections: list[LawSection]




llm = ChatDeepInfra(model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct", max_tokens=100000, temperature=0.6)
#llm = ChatDeepInfra(model_name="google/gemma-3-27b-it", max_tokens=0)
#llm = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct")

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


extract_template = """
Gehe folgende Anweisungen Schrtt für Schritt durch und erstelle jeweils folgende Datenstruktur:
(Operation, Gesetz, 'Paragraph', 'Selector', 'newvalue', 'oldvalue') 

Gesetz ist das Nachweisgesetz.
Paragraph entnimmst du den Änderungsanweisungen (z.B. "§ 1")
Selector steht auch in den Änderungsanweisungen (z.B. "Satz 1" oder "Absatz 1")
Operation: Wähle zwischen "change", "drop", "insert", "replace" 

newvalue ist außer für drop immer notwendig.
oldvalue ist nur bei replace und change zu befüllen, sonst sollst du es weglassen. 
Es ist wichtig, dass du alle Änderungen aufschreibst. Zwischen den Ausgaben sollen keine Aufzählungen oder sonstigen Kommentare sein.
Schneide keine Werte für 'newvalue' oder 'oldvalue' ab!
Vertausche oldvalue und newvalue in keinem Fall. Ordne die Werte richtig zu!

{input}
"""


extract_prompt = ChatPromptTemplate.from_template(extract_template)
extract_chain = extract_prompt | llm





from langchain_core.prompts import PromptTemplate
parser_prompt = PromptTemplate(
    input_variables=["instructions", "completion"],
    template="""
Your task is to fix the output of an LLM so that it matches the expected format.
Do NOT omit any content. Even when the passages are long.
{instructions}
Do only provide the valid JSON, no additional comments before or after it.
Do not fill values with ungiven placeholders.

Original output:
{completion}

Reformatted output:
""".strip()
)

lawparser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=LawText),
    llm=llm,
    max_retries=3,
    prompt=parser_prompt
)

lawchunker = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein Experte im Auftrennen von Gesetzestexten nach Paragraphen. Ändere auf keinen Fall den Inhalt. Kürze nie etwas ab. Schreibe den Inhalt der Abschnitte vollständig ab.."),
    ("user", "{law}")
])

law_parser_chain = lawchunker | llm | lawparser

update_base_template = """
Du bist ein Jura-Experte und aktualisiert Gesetzestexte genau nach der Vorgabe. Du kürzt keine Inhalte.
Füge keine Kommentare vor oder nach dem Inhalt hinzu. Kommentiere deine Arbeit nicht.

Alte Version:
{original}

Anweisung:
"""

change_template = update_base_template + """
Ersetze in {selector} '{oldvalue}' with '{newvalue}'.
Beachte, dass es mehrere passende vorkommen innerhalb des Selektors geben kann.
"""

drop_template = update_base_template + """
Entferne {selector} aus dem ursprünglichen Text.
Denke daran ggf. die Nummerierungen anzupassen.
"""

insert_template = update_base_template + """
Füge nach {selector} folgenden Text ein: {newvalue}
"""

change_chain = ChatPromptTemplate.from_template(change_template) | llm
drop_chain = ChatPromptTemplate.from_template(drop_template) | llm
insert_chain = ChatPromptTemplate.from_template(insert_template) | llm



class AmendmentParserState(TypedDict):
    text: str
    original: str
    amendments: str
    instructions: list[Instruction]
    sections: list[LawSection]



def load_amendments(state: AmendmentParserState) -> AmendmentParserState:
    
    
    response = extract_chain.invoke({"input": state.get("amendments")})
    
    pyparser = PydanticOutputParser(pydantic_object=Instructions)
    parser = OutputFixingParser.from_llm(
        parser=pyparser,
        llm=llm,
        max_retries=3
    )

    fin = parser.invoke(response)
    
    return {**state, "instructions": fin.instructions}


def parse_original_law(state: AmendmentParserState) -> AmendmentParserState:
    with open("data.pkl", "rb") as f:
        sections = pickle.load(f)
        return {
            **state,
            "sections": sections
        }
        
    #res = law_parser_chain.invoke({"law": state.get("original")})
    
    return {
        **state,
        "sections": res.sections
    }

def get_instruction_and_section(state: AmendmentParserState):
    try:
        inst = state.get("instructions")[0]
        section = [section for section in state.get("sections") if section.identifier == inst.paragraph][0]
        return inst, section
    except Exception as e:
        return None, None

def change_passage(state: AmendmentParserState) -> AmendmentParserState:
    sections = state.get("sections")
    inst, section = get_instruction_and_section(state)
    
    if section is not None and inst is not None:
        res = change_chain.invoke({
            "original": section.content,
            "selector": inst.selector,
            "oldvalue": inst.oldvalue,
            "newvalue": inst.newvalue
        })

        section.content = res.content
        sections = [section if s.identifier == section.identifier else s for s in sections]

    return {
        "sections": sections,
        "instructions": state.get("instructions")[1:]
    }

def drop_passage(state: AmendmentParserState) -> AmendmentParserState:
    sections = state.get("sections")
    inst, section = get_instruction_and_section(state)

    if section is not None and inst is not None:
        res = drop_chain.invoke({
            "original": section.content,
            "selector": inst.selector,
        })

        section.content = res.content
        sections = [section if s.identifier == section.identifier else s for s in sections]

    return {
        "sections": sections,
        "instructions": state.get("instructions")[1:]
    }

def insert_passage(state: AmendmentParserState) -> AmendmentParserState:
    sections = state.get("sections")
    inst, section = get_instruction_and_section(state)

    if section is not None and inst is not None:
        res = insert_chain.invoke({
            "original": section.content,
            "selector": inst.selector,
        })
    
        section.content = res.content
        sections = [section if s.identifier == section.identifier else s for s in sections]
    
    return {
        "sections": sections,
        "instructions": state.get("instructions")[1:]
    }

def router(state: AmendmentParserState) -> AmendmentParserState:
    print("instructions")
    return {**state}

def route(state: AmendmentParserState) -> Literal["change", "drop", "insert", END]:
    return state.get("instructions")[0].operation

def empty(state: AmendmentParserState) -> Literal["Empty", "Not Empty"]:
    if len(state.get("instructions", [])) == 0:
        return "Empty"
    return "Not Empty"

def instruction_end(state: AmendmentParserState) -> AmendmentParserState:
    return {**state}

def init(state: AmendmentParserState) -> AmendmentParserState:
    return {**state}


workflow = StateGraph(AmendmentParserState)
workflow.add_node("init", init)
workflow.add_node("load_instructions", load_amendments)
workflow.add_node("load_original", parse_original_law)
workflow.add_node("router", router)
workflow.add_node("change", change_passage)
workflow.add_node("drop", drop_passage)
workflow.add_node("insert", insert_passage)
workflow.add_node("instruction_end", instruction_end)

workflow.add_edge(START, "init")
workflow.add_edge("init", "load_instructions")
workflow.add_edge("load_original", "router")

workflow.add_conditional_edges("load_instructions", empty, {
    "Empty": END,
    "Not Empty": 'load_original'
})

workflow.add_conditional_edges("router", route, {
    "change": "change",
    "drop": "drop",
    "insert": "insert",
})

workflow.add_edge("change", "instruction_end")
workflow.add_edge("drop", "instruction_end")
workflow.add_edge("insert", "instruction_end")

workflow.add_conditional_edges("instruction_end", empty, {
    "Empty": END,
    "Not Empty": "router"
})


app = workflow.compile()

from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
img = app.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
)
display(
    Image(
        img
    )
)

with open("graph.png", "wb") as file:
    file.write(img)

try:
    # Laden der Änderungen aus dem Bundesgesetzblatt
    with open("changes.txt") as file:
        amendments = file.read()

    # Laden der alten Version des Gesetzes
    with open("nachweisg.txt") as file:
        nachweisg = file.read()
except e:
    print("couldnt read file")

def call():
    law = app.invoke({
        "text": nachweisg,
        "amendments": amendments,
        "original": nachweisg,
        "instructions": [],
        "sections": []
    })    



