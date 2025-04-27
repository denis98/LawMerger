# LawMerger
Contribution to Munich Hacking Legal Hackathon

The notebook was used to develop and test nodes from the LangGraph-Workflow step-by-step and was later transfered to graph.py.
Final solution is a Streamlit-App (my first one), which shows the user the current values for the LangGraph-State.
You can also compare the original version with the modified version.

Please install the packages from requirements.txt before running the application.


You can start the application with `streamlit run app.py`.
Please make sure to provide a `DEEPINFRA_API_TOKEN` via .env


The workflow looks like this:
![LangGraph-Workflow](graph.png)

First in load_instructions, the amendment is loaded.
We have provided changes.txt for our presentation case.
When instructions are found, the initial version of the law is loaded.
It can be found in nachweisg.txt.
Then each found instruction is processed. Based on the operation the router calls the specialized node for the task.

Inside the operation-nodes, the matching section is picked and together with the specific instruction a prompt is called.
From invokation of the prompt, we get a new Section-object to replace the original section with.