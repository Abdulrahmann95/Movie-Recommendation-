import streamlit as st
from core import ui

movies, engine, rec = ui.get_movies(), ui.get_engine(), ui.current_rec()
by_id = movies.set_index("id", drop=False)
ss = st.session_state

st.title("💬 Movie chatbot")
st.caption("Offline, rule-based, English only. Type **help** to see what I understand.")

if "chat" not in ss:
    ss.chat = [{"role": "assistant", "ids": [],
                "content": "Hi! 👋 Tell me what you feel like watching. For example: "
                           "*scary movies from the 90s* or *something like Sherlock Jr. but newer*."}]

for i, msg in enumerate(ss.chat):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        ids = [x for x in msg.get("ids", []) if x in by_id.index]
        if ids:
            ui.movie_grid(by_id.loc[ids], key=f"chat{i}", numbered=True)

if prompt := st.chat_input("What do you feel like watching?"):
    last_ids = next((m["ids"] for m in reversed(ss.chat) if m["role"] == "assistant" and m.get("ids")), [])
    reply = engine.handle(prompt, rec, ui.username(), last_ids)
    ss.chat.append({"role": "user", "content": prompt})
    ss.chat.append({"role": "assistant", "content": reply.text, "ids": reply.ids})
    st.rerun()

if len(ss.chat) > 1 and st.button("Clear chat"):
    del ss["chat"]
    st.rerun()
