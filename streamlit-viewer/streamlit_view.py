import streamlit as st
import threading

import osc_server  

def run_osc_server(ip, port):
    osc_server.run_server(ip, port)  

def start_server_once():
    if "server_thread" not in st.session_state:
        thread = threading.Thread(
            target=run_osc_server,
            args=("127.0.0.1", 9000),
            daemon=True
        )
        thread.start()
        st.session_state["server_thread"] = thread
        st.write("OSC Server started on port 9000.")

start_server_once()

st.title("Streamlit Viewer")

if st.button("Show data"):

    slice = osc_server.get_dataframes().get_latest_frames(5)
    data = []

    for frame in slice:
        filtered_data = {k: v for k, v in frame.data.items() if "NeuroFB" in k}
        data.append(filtered_data)


    print(data)
    st.table(data)

