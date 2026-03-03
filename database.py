import requests
import streamlit as st


def save_chat_to_db(username, message, role):
    worker_url = st.secrets["cloudflare"]["worker_url"]
    api_key = st.secrets["cloudflare"]["api_key"]
    url = f"{worker_url}/chat"
    headers = {"X-Custom-Auth": api_key}
    data = {"username": username, "message": message, "role": role}
    requests.post(url, json=data, headers=headers)


def get_chat_history(username):
    worker_url = st.secrets["cloudflare"]["worker_url"]
    api_key = st.secrets["cloudflare"]["api_key"]
    url = f"{worker_url}/history?username={username}"
    headers = {"X-Custom-Auth": api_key}
    response = requests.get(url, headers=headers)
    return response.json() if response.status_code == 200 else []
