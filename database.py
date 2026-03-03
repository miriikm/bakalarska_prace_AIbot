import requests
import streamlit as st


def save_chat_to_db(username, message, role):
    url = f"{st.secrets['CF_WORKER_URL']}/chat"
    headers = {"X-Custom-Auth": st.secrets['CF_API_KEY']}
    data = {"username": username, "message": message, "role": role}
    requests.post(url, json=data, headers=headers)


def get_chat_history(username):
    url = f"{st.secrets['CF_WORKER_URL']}/history?username={username}"
    headers = {"X-Custom-Auth": st.secrets['CF_API_KEY']}
    response = requests.get(url, headers=headers)
    return response.json() if response.status_code == 200 else []
