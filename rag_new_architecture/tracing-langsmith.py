from langchain.messages import HumanMessage
import uuid
from graph import graph
import pandas as pd

df_testing = pd.read_excel("Testing ChatAI Valak.xlsx", skiprows=1, header=0)
pertanyaan_list = df_testing.iloc[:, 3].dropna().tolist()

pertanyaan_list

for i in pertanyaan_list:
    thread_id = str(uuid.uuid4())
    final_state = graph.invoke(
        {"messages": [HumanMessage(content=i)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print(f"✅ Thread ID: {thread_id} done")
    print(final_state)
    print("-" * 40)