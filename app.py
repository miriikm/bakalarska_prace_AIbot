import streamlit as st
import os
import shutil
import json
import yaml
import secrets
from datetime import datetime
from pathlib import Path
from pydantic import SecretStr
from database import get_chat_history, save_chat_to_db

from dotenv import load_dotenv

import streamlit_authenticator as stauth

from langchain_community.document_loaders import RecursiveUrlLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

os.environ["GOOGLE_API_VERSION"] = "v1"

# --- KONFIGURACE ---
_SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(_SCRIPT_DIR / ".env")
DOCS_DIR = _SCRIPT_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)
FAISS_INDEX_STATIC = _SCRIPT_DIR / "faiss_index_static"
FAISS_INDEX_DIR = _SCRIPT_DIR / "faiss_index_local"
HISTORY_FILE = _SCRIPT_DIR / "history.json"
CONFIG_FILE = _SCRIPT_DIR / "config.yaml"

def _env_api_key(provider: str) -> str:
    if provider in ("Google Gemini", "Gemini 2.0 Flash"):
        return (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    if provider == "OpenAI (ChatGPT)":
        return (os.environ.get("OPENAI_API_KEY") or "").strip()
    if provider == "GPT-4o (GitHub)":
        return (os.environ.get("GITHUB_TOKEN") or "").strip()
    return ""

def _get_global_api_key(provider: str) -> str:
    from_env = _env_api_key(provider)
    if from_env:
        return from_env
    try:
        if hasattr(st, "secrets"):
            if provider in ("Google Gemini", "Gemini 2.0 Flash"):
                return (st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("gemini_api_key") or "").strip()
            if provider == "OpenAI (ChatGPT)":
                return (st.secrets.get("OPENAI_API_KEY") or "").strip()
            if provider == "GPT-4o (GitHub)":
                return (st.secrets.get("GITHUB_TOKEN") or "").strip()
    except Exception:
        pass
    return ""

def _get_default_gemini_api_key() -> str:
    from_env = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("RADSEQ_DEFAULT_GEMINI_API_KEY") or "").strip()
    if from_env:
        return from_env
    try:
        if hasattr(st, "secrets") and st.secrets.get("gemini_api_key"):
            return str(st.secrets["gemini_api_key"]).strip()
    except Exception:
        pass
    return ""

def _resolve_api_key(provider: str, user_key: str) -> tuple[str, bool]:
    global_key = _get_global_api_key(provider)
    user_val = (user_key or "").strip()
    if user_val:
        return user_val, False
    if global_key:
        return global_key, True
    if provider in ("Google Gemini", "Gemini 2.0 Flash"):
        effective = _get_default_gemini_api_key()
        return effective, bool(effective)
    return "", False


def get_github_model_response(prompt, context):
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=st.secrets["GITHUB_TOKEN"],
    )
    sys = "Jsi vědecký asistent. Při odpovídání rozlišuj původ informací: pokud informaci čerpáš z poskytnutého kontextu (manuálu), vlož za větu nebo odstavec značku [MANUAL]; pokud ze svých znalostí, vlož [AI]. Příklad: PCA analýza slouží k vizualizaci genetických struktur [MANUAL]. Je to jedna z nejpoužívanějších metod v bioinformatice [AI]."
    if not (context or "").strip():
        sys += " Nemáš k dispozici kontext z manuálu, označuj vše jako [AI]."
    full_prompt = f"Kontext z manuálů: {context}\n\nOtázka: {prompt}"
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": full_prompt}
        ],
        model="gpt-4o",
        temperature=0.2
    )
    return response.choices[0].message.content

# --- SPRÁVA KONFIGURACE A AUTENTIZACE ---
def _default_config() -> dict:
    return {
        "credentials": {"usernames": {}},
        "cookie": {
            "expiry_days": 0,
            "key": secrets.token_hex(16),
            "name": "radseq_auth"
        },
        "preauthorized": {"emails": []}
    }

def load_config() -> dict:
    config_path = str(CONFIG_FILE.resolve())
    if not CONFIG_FILE.exists():
        default_config = _default_config()
        save_config(default_config)
        return default_config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data or not isinstance(data, dict):
            return _default_config()
        if "credentials" not in data:
            data["credentials"] = {"usernames": {}}
        if "cookie" not in data:
            data["cookie"] = {"expiry_days": 0, "key": secrets.token_hex(16), "name": "radseq_auth"}
        if "preauthorized" not in data:
            data["preauthorized"] = {"emails": []}
        return data
    except Exception:
        default_config = _default_config()
        save_config(default_config)
        return default_config

def save_config(config: dict):
    config_path = str(CONFIG_FILE.resolve())
    tmp_path = config_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        if os.path.exists(config_path):
            backup_path = config_path + ".backup"
            try:
                shutil.copy2(config_path, backup_path)
            except Exception:
                pass
        os.replace(tmp_path, config_path)
    except Exception:
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        except Exception:
            raise

def get_user_email_by_username(username: str, config: dict) -> str:
    """Získá email uživatele podle username."""
    usernames = config.get("credentials", {}).get("usernames", {})
    if username in usernames:
        return usernames[username].get("email", "")
    return ""

def get_username_by_email(email: str, config: dict) -> str:
    """Získá username podle emailu."""
    if not email:
        return ""
    usernames = config.get("credentials", {}).get("usernames", {})
    email_lower = str(email).lower() if email else ""
    for uname, udata in usernames.items():
        user_email = udata.get("email", "")
        if user_email and str(user_email).lower() == email_lower:
            return uname
    return ""

# --- SPRÁVA HISTORIE KONVERZACÍ ---
def load_history(username: str) -> dict:
    """Načte historii konverzací pro daného uživatele (podle username)."""
    if not username:
        return {}
    if not HISTORY_FILE.exists():
        return {}
    
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            all_history = json.load(f)
        return all_history.get(username, {}) if username else {}
    except Exception:
        return {}

def save_history(username: str, conversation_id: str, title: str, messages: list, timestamp: str):
    """Uloží nebo aktualizuje konverzaci do historie (podle username)."""
    if not username:
        return
    
    all_history = {}
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                all_history = json.load(f)
        except Exception:
            pass
    
    if username not in all_history:
        all_history[username] = {}
    
    # CHYTRÁ AKTUALIZACE: Pokud existuje stejné ID, aktualizujeme existující záznam
    # Jinak vytvoříme nový záznam
    all_history[username][conversation_id] = {
        "id": conversation_id,
        "timestamp": timestamp,
        "title": title,
        "messages": messages
    }
    
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_history, f, ensure_ascii=False, indent=2)

def delete_conversation(username: str, conversation_id: str) -> bool:
    """Smaže konverzaci z historie pro daného uživatele. Vrací True, pokud byla smazána."""
    if not username or not conversation_id:
        return False
    
    if not HISTORY_FILE.exists():
        return False
    
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            all_history = json.load(f)
        
        if username not in all_history:
            return False
        
        if conversation_id not in all_history[username]:
            return False
        
        # Smazání konverzace
        del all_history[username][conversation_id]
        
        # Uložení aktualizované historie
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(all_history, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception:
        return False

def load_manual_from_url(url: str) -> tuple[bool, str, int]:
    """
    Načte manuál z URL a uloží ho do docs/.
    Vrací tuple: (success: bool, filename: str, num_docs: int)
    """
    if not url or not isinstance(url, str):
        return False, "", 0
    
    # Validace URL
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return False, "", 0
    
    try:
        # Načtení dokumentů z URL pomocí RecursiveUrlLoader
        loader = RecursiveUrlLoader(url=url, max_depth=2, timeout=10)
        docs = loader.load()
        
        if not docs or len(docs) == 0:
            return False, "", 0
        
        # Vytvoření názvu souboru s hash z URL
        url_hash = secrets.token_hex(4)
        # Získání domény z URL pro lepší název souboru
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace("www.", "").replace(".", "_")
            filename = f"web_manual_{domain}_{url_hash}.txt"
        except:
            filename = f"web_manual_{url_hash}.txt"
        
        # Kombinace obsahu všech dokumentů
        combined_content = []
        for doc in docs:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            source = doc.metadata.get('source', url) if hasattr(doc, 'metadata') else url
            combined_content.append(f"=== Zdroj: {source} ===\n\n{content}\n\n")
        
        # Uložení do souboru
        save_path = DOCS_DIR / filename
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(combined_content))
        
        return True, filename, len(docs)
    except Exception as e:
        return False, str(e), 0


def get_authenticator(config: dict):
    """Vytvoří a vrátí authenticator."""
    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )

# --- AUTENTIZACE A REGISTRACE ---
def show_login_registration_tabs(config: dict, authenticator):
    """Zobrazí taby pro přihlášení a registraci."""
    tab1, tab2 = st.tabs(["Přihlášení", "Registrace"])
    
    with tab1:
        st.subheader("Přihlášení")
        try:
            login_result = authenticator.login(location="main")
            if login_result and len(login_result) == 3:
                name, authentication_status, username = login_result
                
                if authentication_status:
                    if username:
                        st.session_state["authentication_status"] = True
                        st.session_state["name"] = name
                        st.session_state["username"] = username
                        user_email = get_user_email_by_username(username, config)
                        st.session_state["user_email"] = user_email
                        st.success(f"Vítejte, {name}!")
                        st.rerun()
                elif authentication_status == False:
                    st.error("Nesprávné uživatelské jméno nebo heslo.")
                elif authentication_status == None:
                    st.warning("Zadejte prosím své přihlašovací údaje.")
        except Exception as e:
            st.warning("Zadejte prosím své přihlašovací údaje.")
    
    with tab2:
        st.subheader("Registrace")
        with st.form("registration_form"):
            new_username = st.text_input("Uživatelské jméno *")
            new_name = st.text_input("Celé jméno")
            new_email = st.text_input("E-mail *")
            new_password = st.text_input("Heslo (min. 4 znaky) *", type="password")
            confirm_password = st.text_input("Potvrzení hesla *", type="password")
            submitted = st.form_submit_button("Registrovat", type="primary")
        
        if submitted:
            usernames = config.get("credentials", {}).get("usernames", {})
            
            # Validace
            if not new_username or not new_email or not new_password:
                st.error("Vyplňte prosím všechny povinné údaje (označené *).")
            elif len(new_password) < 4:
                st.error("Heslo musí mít alespoň 4 znaky.")
            elif new_password != confirm_password:
                st.error("Hesla se neshodují.")
            elif new_username in usernames:
                st.error("Toto uživatelské jméno je již obsazeno.")
            elif any(str(u.get("email", "")).lower() == str(new_email).lower() if u.get("email") and new_email else False for u in usernames.values()):
                st.error("Tento e-mail je již použit u jiného účtu.")
            else:
                # Hashování hesla pomocí bcrypt
                import bcrypt
                hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                
                usernames[new_username] = {
                    "name": new_name or new_username,
                    "email": str(new_email).lower() if new_email else "",
                    "password": hashed_password,
                }
                config["credentials"]["usernames"] = usernames
                save_config(config)
                st.session_state["authentication_status"] = True
                st.session_state["username"] = new_username
                st.session_state["name"] = new_name or new_username
                st.session_state["user_email"] = str(new_email).strip().lower() if new_email else ""
                st.success("Účet byl vytvořen. Jste přihlášeni.")
                st.rerun()

def show_profile_management(config: dict, username: str):
    """Zobrazí sekci pro správu profilu."""
    with st.expander("👤 Můj profil", expanded=False):
        usernames = config.get("credentials", {}).get("usernames", {})
        if username not in usernames:
            st.error("Uživatel nenalezen.")
            return
        
        user_data = usernames[username]
        current_email = user_data.get("email", "")
        current_name = user_data.get("name", "")
        
        st.write(f"**Jméno:** {current_name}")
        st.write(f"**E-mail:** {current_email}")
        st.write(f"**Uživatelské jméno:** {username}")
        
        st.divider()
        
        # Změna uživatelského jména
        st.subheader("Změna uživatelského jména")
        with st.form("change_username_form"):
            new_username = st.text_input("Nové uživatelské jméno", value=username)
            change_username_submitted = st.form_submit_button("Změnit uživatelské jméno")
        
        if change_username_submitted:
            if new_username == username:
                st.info("Zadali jste stejné uživatelské jméno.")
            elif new_username in usernames:
                st.error("Toto uživatelské jméno je již obsazeno.")
            elif not new_username:
                st.error("Uživatelské jméno nemůže být prázdné.")
            else:
                # Migrace historie z jednoho username na druhý
                if HISTORY_FILE.exists():
                    try:
                        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                            all_history = json.load(f)
                        
                        # Pokud existuje historie pod starým username, přesuneme ji na nový
                        if username in all_history:
                            all_history[new_username] = all_history.pop(username)
                            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                                json.dump(all_history, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass  # Pokud migrace selže, pokračujeme dál
                
                # Aktualizace config
                usernames[new_username] = usernames.pop(username)
                config["credentials"]["usernames"] = usernames
                save_config(config)
                
                st.session_state["username"] = new_username
                st.success("Uživatelské jméno bylo úspěšně změněno!")
                st.rerun()
        
        st.divider()
        
        # Správa Google API klíče
        st.subheader("Google API klíč")
        stored_api_key = user_data.get("api_key", "")
        with st.form("save_api_key_form"):
            api_key_input = st.text_input(
                "Google API Key", 
                value=stored_api_key if stored_api_key else "",
                type="password",
                help="Zadejte svůj Google API klíč pro Gemini. Klíč bude uložen do vašeho profilu."
            )
            save_api_key_submitted = st.form_submit_button("💾 Uložit API klíč")
        
        if save_api_key_submitted:
            if api_key_input:
                # Uložení API klíče do profilu uživatele
                usernames[username]["api_key"] = api_key_input
                config["credentials"]["usernames"] = usernames
                save_config(config)
                st.success("API klíč byl úspěšně uložen do vašeho profilu!")
                st.rerun()
            else:
                st.error("Zadejte prosím API klíč.")
        
        if stored_api_key:
            st.info("✅ API klíč je uložen v profilu. Klíč bude automaticky použit při chatování.")
        
        st.divider()
        
        # Změna hesla - vyžaduje staré heslo, nové heslo a potvrzení
        st.subheader("Změna hesla")
        with st.form("change_password_form"):
            old_password = st.text_input("Současné heslo", type="password")
            new_password = st.text_input("Nové heslo (min. 4 znaky)", type="password")
            confirm_new_password = st.text_input("Potvrzení nového hesla", type="password")
            change_password_submitted = st.form_submit_button("Změnit heslo")
        
        if change_password_submitted:
            if not old_password or not new_password or not confirm_new_password:
                st.error("Vyplňte prosím všechna pole.")
            elif len(new_password) < 4:
                st.error("Nové heslo musí mít alespoň 4 znaky.")
            elif new_password != confirm_new_password:
                st.error("Nová hesla se neshodují.")
            else:
                # Ověření starého hesla pomocí bcrypt
                try:
                    import bcrypt
                    stored_hash = usernames[username]["password"]
                    if bcrypt.checkpw(old_password.encode('utf-8'), stored_hash.encode('utf-8')):
                        # Heslo je správné, uložíme nové pomocí bcrypt
                        hashed_new_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                        
                        usernames[username]["password"] = hashed_new_password
                        config["credentials"]["usernames"] = usernames
                        save_config(config)
                        st.success("Heslo bylo úspěšně změněno!")
                        st.rerun()
                    else:
                        st.error("Současné heslo je nesprávné.")
                except Exception as err:
                    st.error(f"Chyba při ověřování hesla: {err}")

GEMINI_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

def get_gemini_model(api_key: str):
    key = (api_key or "").strip() or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        return None
    try:
        genai.configure(api_key=key)  # type: ignore[attr-defined]
        for model_name in ("models/gemini-2.5-flash", "models/gemini-1.5-flash", "gemini-1.5-flash"):
            try:
                return genai.GenerativeModel(model_name=model_name)  # type: ignore[attr-defined]
            except Exception:
                continue
        return None
    except Exception:
        return None

def get_gemini_response(model, full_prompt: str) -> str:
    try:
        response = model.generate_content(full_prompt, safety_settings=GEMINI_SAFETY_SETTINGS)
        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        return "Model vrátil prázdnou odpověď."
    except Exception as e:
        st.error(f"Technická chyba API: {str(e)}")
        return ""

def create_llm_instance(provider: str, api_key: str):
    if not api_key or api_key is None or (isinstance(api_key, str) and api_key.strip() == ""):
        return None
    try:
        if provider == "OpenAI (ChatGPT)":
            return ChatOpenAI(model="gpt-4o-mini", api_key=SecretStr(api_key), temperature=0.7)
        return None
    except Exception as e:
        st.error(f"Chyba při inicializaci {provider}: {e}")
        return None

def generate_conversation_title(first_question: str, api_key: str, provider: str = "Google Gemini") -> str:
    words = first_question.strip().split()[:5]
    return " ".join(words) if words else f"Konverzace {datetime.now().strftime('%d.%m')}"

def _format_citations(text: str) -> str:
    if not text:
        return text
    return text.replace("[MANUAL]", " 📖").replace("[AI]", " 🤖")

PROMPT_TEMPLATE = """Jsi vědecký asistent. Při odpovídání rozlišuj původ informací:
- Pokud informaci čerpáš z poskytnutého kontextu (manuálu), vlož za danou větu nebo odstavec značku [MANUAL].
- Pokud informaci doplňuješ ze svých obecných znalostí, vlož značku [AI].
Příklad: PCA analýza slouží k vizualizaci genetických struktur [MANUAL]. Je to jedna z nejpoužívanějších metod v bioinformatice [AI].

KONTEXT Z MANUÁLU:
{context}

Otázka: {question}"""

PROMPT_TEMPLATE_NO_CONTEXT = """Jsi vědecký asistent. Nemáš k dispozici kontext z manuálu, označuj vše jako [AI].
Uživatel se ptá na: {question}. Odpověz ze svých obecných znalostí o genetice a u každé věty/odstavce použij [AI]."""

def classify_user_intent(prompt: str, api_key: str, provider: str = "Google Gemini") -> str:
    greeting_keywords = ["ahoj", "čau", "dobrý den", "dobrý večer", "děkuji", "děkuju", "díky",
                         "co umíš", "co dokážeš", "pomoc", "help", "hello", "hi", "thanks", "thank you"]
    prompt_lower = str(prompt).lower().strip()
    if len(prompt_lower.split()) <= 5:
        for keyword in greeting_keywords:
            if keyword in prompt_lower:
                return "greeting"
    return "technical"

def _get_local_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None

def _get_embeddings(api_key: str, provider: str):
    if not api_key or not api_key.strip():
        return None
    try:
        if provider == "OpenAI (ChatGPT)":
            return OpenAIEmbeddings(api_key=SecretStr(api_key), model="text-embedding-3-small")
    except Exception:
        return None
    return None

def _stream_or_invoke(llm, prompt: str) -> tuple[str, bool]:
    try:
        if getattr(llm, "streaming", False) and hasattr(llm, "stream"):
            def _gen():
                for chunk in llm.stream(prompt):
                    if getattr(chunk, "content", None):
                        yield chunk.content
            content = st.write_stream(_gen())
            return (str(content) if content else "").strip(), True
    except Exception:
        pass
    r = llm.invoke(prompt)
    content = (getattr(r, "content", None) or str(r) or "").strip()
    return content, False

def _get_rag_failure_reason() -> str:
    if not FAISS_INDEX_STATIC.exists():
        return "Složka faiss_index_static v projektu chybí. Nahraj ji na hosting (včetně index.faiss a index.pkl)."
    if not (FAISS_INDEX_STATIC / "index.faiss").exists():
        return "V faiss_index_static chybí soubor index.faiss. Spusť lokálně: python build_faiss_from_static.py a nahraj složku na git."
    emb = _get_local_embeddings()
    if emb is None:
        return "Embedding model (sentence-transformers) se na tomto prostředí nepodařilo načíst. Na hostingu může chybět závislost nebo síťové stažení modelu."
    for index_dir in (FAISS_INDEX_STATIC, FAISS_INDEX_DIR):
        if not index_dir.exists() or not (index_dir / "index.faiss").exists():
            continue
        try:
            FAISS.load_local(str(index_dir), emb, allow_dangerous_deserialization=True)
            return ""
        except Exception as e:
            return f"Index se na tomto prostředí nepodařilo načíst (např. jiná platforma než při sestavení). Chyba: {type(e).__name__}: {e}"
    return "Index nebyl nalezen ani v faiss_index_static, ani v faiss_index_local."

@st.cache_resource(show_spinner=False)
def get_vectorstore_for_query(_api_key: str, _provider: str):
    if not FAISS_INDEX_STATIC.exists():
        return None
    if not (FAISS_INDEX_STATIC / "index.faiss").exists():
        return None
    embeddings = _get_local_embeddings()
    if embeddings is None:
        return None
    for index_dir in (FAISS_INDEX_STATIC, FAISS_INDEX_DIR):
        if not index_dir.exists() or not (index_dir / "index.faiss").exists():
            continue
        try:
            return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
        except Exception:
            continue
    return None

@st.cache_resource(show_spinner=True)
def build_vectorstore(_api_key: str, _provider: str):
    embeddings = _get_local_embeddings()
    if embeddings is None:
        return None
    index_path = str(FAISS_INDEX_STATIC)
    if FAISS_INDEX_STATIC.exists() and (FAISS_INDEX_STATIC / "index.faiss").exists():
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            pass
    all_docs = []
    if DOCS_DIR.exists() and any(DOCS_DIR.iterdir()):
        for pattern in ["**/*.md", "**/*.txt", "**/*.pdf"]:
            try:
                local_loader = DirectoryLoader(str(DOCS_DIR), glob=pattern)
                all_docs.extend(local_loader.load())
            except Exception:
                continue
    if not all_docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(all_docs)
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
    except Exception:
        return None
    vectorstore.save_local(index_path)
    return vectorstore

def main():
    st.set_page_config(page_title="RAD-seq Bioinfo Helper", layout="wide")
    if "_cache_cleared_at_start" not in st.session_state:
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        st.session_state["_cache_cleared_at_start"] = True
    config = load_config()
    
    # Vytvoření authenticatoru JEDNOU pomocí cache (aby se předešlo duplicitním klíčům)
    authenticator = get_authenticator(config)
    
    # Inicializace session state pro autentizaci
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = None
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "name" not in st.session_state:
        st.session_state["name"] = None
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = None
    
    # Pokud není uživatel přihlášen, zobrazíme přihlášení/registraci
    if not st.session_state.get("authentication_status"):
        st.title("🧬 RAD-seq Asistent")
        show_login_registration_tabs(config, authenticator)
        return
    
    # Uživatel je přihlášen - zobrazíme chatbot UI
    username = st.session_state["username"]
    user_email = st.session_state["user_email"]
    name = st.session_state["name"]
    
    # Načtení API klíče a poskytovatele z profilu uživatele
    usernames = config.get("credentials", {}).get("usernames", {})
    stored_api_key = usernames.get(username, {}).get("api_key", "") if username in usernames else ""
    stored_provider = usernames.get(username, {}).get("ai_provider", "Gemini 2.0 Flash") if username in usernames else "Gemini 2.0 Flash"
    
    # Inicializace session state pro zprávy a aktuální konverzaci
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "current_conversation_id" not in st.session_state:
        st.session_state["current_conversation_id"] = None
    if "conversation_title" not in st.session_state:
        st.session_state["conversation_title"] = None
    
    # Načtení historie pro uživatele (podle username)
    user_history = load_history(username)

    with st.sidebar:
        # HORNÍ ČÁST SIDEBARU - Info o uživateli a hlavní akce
        st.markdown(f"**Přihlášen:** {name or username}")
        st.caption(f"@{username}")
        
        # Zobrazení aktivního AI modelu
        current_provider = st.session_state.get("ai_provider", stored_provider)
        if current_provider not in ("Gemini 2.0 Flash", "GPT-4o (GitHub)"):
            st.session_state["ai_provider"] = "Gemini 2.0 Flash"
            current_provider = "Gemini 2.0 Flash"
        provider_icon = "🤖" if current_provider == "Gemini 2.0 Flash" else "💬"
        st.caption(f"{provider_icon} **Aktivní model:** {current_provider}")
        _uk = st.session_state.get("api_key_input", "") or stored_api_key
        _eff, _from_secrets = _resolve_api_key(current_provider, _uk)
        if current_provider == "GPT-4o (GitHub)":
            st.caption("Token: z secrets" if _eff else "GitHub token: nezadán")
        elif not _eff:
            st.caption("API klíč: nezadán")
        else:
            st.caption("API klíč: z Nastavení" if not _from_secrets else ("API klíč: z .env" if (_eff and _env_api_key(current_provider) == _eff) else "API klíč: z secrets/prostředí"))
        st.divider()
        
        # Tlačítko "Nová konverzace"
        if st.button("➕ Nová konverzace", use_container_width=True, type="primary"):
            st.session_state["messages"] = []
            st.session_state["current_conversation_id"] = None
            st.session_state["conversation_title"] = None
            st.rerun()
        
        if st.button("🔄 Nouzový reset aplikace", use_container_width=True, help="Vymaže session a restartuje aplikaci při zaseknutí."):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Logout tlačítko
        try:
            authenticator.logout('🚪 Odhlásit se', 'sidebar')
            # Pokud authenticator.logout() nastavil logout flag, vyčistíme session_state
            if st.session_state.get("logout"):
                # Úplné vymazání session_state - kompletní vyčištění relace
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            # Kontrola po logout - pokud není přihlášen, přesměrujeme na úvodní obrazovku
            if not st.session_state.get("authentication_status"):
                st.rerun()
        except Exception:
            # Pokud authenticator.logout() způsobí chybu, použijeme vlastní tlačítko
            if st.button("🚪 Odhlásit se", use_container_width=True, type="secondary", key="logout_button"):
                # Úplné vymazání session_state - kompletní vyčištění relace
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Kontrola po logout - pokud není přihlášen, přesměrujeme na úvodní obrazovku
        if not st.session_state.get("authentication_status"):
            st.rerun()

        st.divider()
        
        # SEKCE HISTORIE - Expander s historií konverzací
        with st.expander("📜 Historie konverzací", expanded=False):
            if user_history:
                # Seřadíme podle časového razítka (nejnovější první)
                sorted_conversations = sorted(
                    user_history.items(),
                    key=lambda x: x[1].get("timestamp", ""),
                    reverse=True
                )
                
                for conv_id, conv_data in sorted_conversations:
                    title = conv_data.get("title", "Bez názvu")
                    timestamp = conv_data.get("timestamp", "")
                    # Formátování data pro zobrazení
                    try:
                        if timestamp:
                            dt = datetime.fromisoformat(timestamp)
                            date_str = dt.strftime("%d.%m.%Y %H:%M")
                        else:
                            date_str = "Neznámé datum"
                    except:
                        date_str = timestamp[:10] if len(timestamp) >= 10 else timestamp
                    
                    # Vytvoříme kontejner s dvěma sloupci: tlačítko pro načtení a tlačítko pro smazání
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        # Tlačítko pro načtení chatu
                        button_label = f"{title}\n📅 {date_str}"
                        if st.button(button_label, key=f"hist_{conv_id}", use_container_width=True):
                            # Načteme konverzaci do session_state
                            st.session_state["messages"] = conv_data.get("messages", [])
                            st.session_state["current_conversation_id"] = conv_id
                            st.session_state["conversation_title"] = title
                            st.rerun()
                    
                    with col2:
                        # Tlačítko pro smazání s potvrzením
                        delete_key = f"delete_{conv_id}"
                        confirm_key = f"confirm_delete_{conv_id}"
                        
                        # Zkontrolujeme, zda uživatel potvrdil smazání
                        if st.session_state.get(confirm_key, False):
                            # Zobrazíme varování a tlačítko pro potvrzení
                            st.warning("⚠️ Smazat?")
                            if st.button("✅ Ano", key=f"yes_{conv_id}", use_container_width=True):
                                # Smazání konverzace
                                if delete_conversation(username, conv_id):
                                    # Pokud je to aktuálně otevřený chat, vyčistíme session_state
                                    if st.session_state.get("current_conversation_id") == conv_id:
                                        st.session_state["messages"] = []
                                        st.session_state["current_conversation_id"] = None
                                        st.session_state["conversation_title"] = None
                                    
                                    # Vyčistíme potvrzovací flag
                                    if confirm_key in st.session_state:
                                        del st.session_state[confirm_key]
                                    
                                    st.success("Konverzace byla smazána.")
                                    st.rerun()
                                else:
                                    st.error("Nepodařilo se smazat konverzaci.")
                            
                            if st.button("❌ Ne", key=f"no_{conv_id}", use_container_width=True):
                                # Zrušíme potvrzení
                                if confirm_key in st.session_state:
                                    del st.session_state[confirm_key]
                                st.rerun()
                        else:
                            # Zobrazíme tlačítko pro smazání
                            if st.button("🗑️", key=delete_key, use_container_width=True, help="Smazat tuto konverzaci"):
                                # Nastavíme potvrzovací flag
                                st.session_state[confirm_key] = True
                                st.rerun()
            else:
                st.info("Zatím žádné uložené konverzace.")
        
        with st.expander("📊 Debug: kontext z manuálu", expanded=False):
            _n = st.session_state.get("_last_context_length")
            if _n is not None:
                st.caption(f"Délka kontextu z manuálu (poslední odpověď): **{_n}** znaků")
            else:
                st.caption("Zatím žádná data (pošlete zprávu pro zobrazení).")
        
        # SEKCE MANUÁLY - Expander se správou manuálů
        with st.expander("📚 Správa manuálů", expanded=False):
            uploaded_file = st.file_uploader(
                "Nahraj manuál (.md, .txt nebo .pdf)",
                type=["md", "txt", "pdf"],
            )

            if uploaded_file is not None:
                save_path = DOCS_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Soubor '{uploaded_file.name}' byl uložen do složky 'docs/'.")

            st.divider()
            
            # Načtení manuálu z URL
            st.subheader("🌐 Načtení manuálu z webu")
            url_input = st.text_input(
                "Vložit odkaz na manuál (URL)",
                placeholder="https://example.com/manual",
                help="Zadejte URL adresu manuálu, který chcete přidat do znalostní báze. Načte se obsah z dané stránky a podstránek (max. hloubka 2)."
            )
            
            if st.button("📥 Načíst manuál z webu", use_container_width=True):
                if not url_input or not url_input.strip():
                    st.error("⚠️ Prosím, zadejte platnou URL adresu.")
                else:
                    # Validace URL
                    url = url_input.strip()
                    if not url.startswith(("http://", "https://")):
                        st.error("⚠️ URL musí začínat na 'http://' nebo 'https://'.")
                    else:
                        # Načtení manuálu z URL
                        with st.spinner("Stahuji a indexuji webový manuál..."):
                            success, result, num_docs = load_manual_from_url(url)
                            
                            if success:
                                st.success(f"✅ Manuál byl úspěšně stažen z '{url}' a uložen jako '{result}' ({num_docs} stránek).")
                                
                                # Automatické spuštění rebuild znalostní báze
                                with st.spinner("Přestavuji znalostní bázi s novým manuálem..."):
                                    if FAISS_INDEX_DIR.exists():
                                        shutil.rmtree(str(FAISS_INDEX_DIR))
                                    st.cache_resource.clear()
                                    _key, _ = _resolve_api_key(st.session_state.get("ai_provider", stored_provider), st.session_state.get("api_key_input", "") or stored_api_key)
                                    vs = build_vectorstore(_key, st.session_state.get("ai_provider", stored_provider))
                                
                                if vs is not None:
                                    st.success("🎉 Znalostní báze byla úspěšně aktualizována a nový webový manuál je nyní k dispozici v chatu!")
                                else:
                                    st.warning("Manuál byl stažen, ale znalostní bázi se nepodařilo znovu vytvořit. Zkuste kliknout na 'Rebuild Knowledge Base' níže.")
                            else:
                                # result obsahuje chybovou zprávu
                                error_msg = result if result else "Neznámá chyba"
                                st.error(f"❌ Nepodařilo se načíst manuál z URL '{url}': {error_msg}")
                                st.info("💡 Tip: Některé weby mohou blokovat automatické stahování. Zkuste jinou URL nebo nahrajte manuál jako soubor.")

            st.divider()
            
            # Rebuild znalostní báze
            if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
                with st.spinner("Přestavuji znalostní bázi..."):
                    if FAISS_INDEX_DIR.exists():
                        shutil.rmtree(str(FAISS_INDEX_DIR))
                    st.cache_resource.clear()
                    _key, _ = _resolve_api_key(st.session_state.get("ai_provider", stored_provider), st.session_state.get("api_key_input", "") or stored_api_key)
                    vs = build_vectorstore(_key, st.session_state.get("ai_provider", stored_provider))

                if vs is not None:
                    st.success("Znalostní báze byla úspěšně znovu vytvořena a nový manuál byl integrován.")
                else:
                    st.warning("Znalostní bázi se nepodařilo znovu vytvořit – nebyla nalezena žádná data k indexaci.")
        
        # SEKCE NASTAVENÍ A PROFIL - Expander s nastavením
        # Inicializace session_state pro api_key_input a provider
        if "api_key_input" not in st.session_state:
            st.session_state["api_key_input"] = stored_api_key if stored_api_key else ""
        if "ai_provider" not in st.session_state:
            st.session_state["ai_provider"] = stored_provider
        
        with st.expander("⚙️ Nastavení a Profil", expanded=False):
            # Výběr poskytovatele AI
            provider_options = ["Gemini 2.0 Flash", "GPT-4o (GitHub)"]
            selected_provider = st.selectbox(
                "Vyberte poskytovatele AI",
                options=provider_options,
                index=provider_options.index(st.session_state["ai_provider"]) if st.session_state["ai_provider"] in provider_options else 0,
                help="Vyberte, který AI model chcete používat pro generování odpovědí."
            )
            
            # Uložení volby poskytovatele do session_state a config.yaml
            if selected_provider != st.session_state.get("ai_provider", stored_provider):
                st.session_state["ai_provider"] = selected_provider
                # Uložení do config.yaml
                if username in usernames:
                    usernames[username]["ai_provider"] = selected_provider
                    config["credentials"]["usernames"] = usernames
                    save_config(config)
                    st.success(f"✅ Poskytovatel změněn na {selected_provider}")
                    st.rerun()
            
            st.divider()
            
            _uk_inner = st.session_state.get("api_key_input", "") or stored_api_key
            _eff_key, using_global_key = _resolve_api_key(selected_provider, _uk_inner)
            api_key_input = _uk_inner
            if selected_provider == "GPT-4o (GitHub)":
                if _eff_key:
                    st.info("Používá se GitHub token z .streamlit/secrets.toml (GITHUB_TOKEN).")
                else:
                    st.warning("⚠️ Pro GPT-4o (GitHub) nastavte GITHUB_TOKEN v .streamlit/secrets.toml.")
            else:
                if using_global_key and _eff_key:
                    st.info("Aktuálně se používá globální API klíč administrátora.")
                api_key_label = "Google API Key" if selected_provider == "Gemini 2.0 Flash" else "OpenAI API Key"
                api_key_help = f"Zadejte nebo upravte svůj {api_key_label}. Uloží se do config.yaml do vašeho profilu a zůstane i po odhlášení."
                if stored_api_key and not using_global_key:
                    st.caption(f"✅ V profilu máte uložen vlastní API klíč pro {selected_provider}.")
                api_key_input = st.text_input(
                    api_key_label,
                    value=st.session_state["api_key_input"],
                    type="password",
                    help=api_key_help,
                    key="api_key_input_field"
                )
                if api_key_input:
                    st.session_state["api_key_input"] = api_key_input
            if selected_provider != "GPT-4o (GitHub)" and st.session_state.get("api_key_input") != stored_api_key:
                if st.button("💾 Uložit API klíč do profilu", use_container_width=True):
                    usernames[username]["api_key"] = api_key_input
                    config["credentials"]["usernames"] = usernames
                    save_config(config)
                    st.session_state["api_key_input"] = api_key_input
                    st.success("API klíč byl úspěšně uložen do vašeho profilu!")
                    st.rerun()
            
            st.divider()
            
            # Změna uživatelského jména
            usernames = config.get("credentials", {}).get("usernames", {})
            if username in usernames:
                user_data = usernames[username]
                current_email = user_data.get("email", "")
                current_name = user_data.get("name", "")
                
                st.subheader("Změna uživatelského jména")
                with st.form("change_username_form"):
                    new_username = st.text_input("Nové uživatelské jméno", value=username)
                    change_username_submitted = st.form_submit_button("Změnit uživatelské jméno", use_container_width=True)
                
                if change_username_submitted:
                    if new_username == username:
                        st.info("Zadali jste stejné uživatelské jméno.")
                    elif new_username in usernames:
                        st.error("Toto uživatelské jméno je již obsazeno.")
                    elif not new_username:
                        st.error("Uživatelské jméno nemůže být prázdné.")
                    else:
                        # Migrace historie z jednoho username na druhý
                        if HISTORY_FILE.exists():
                            try:
                                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                                    all_history = json.load(f)
                                
                                # Pokud existuje historie pod starým username, přesuneme ji na nový
                                if username in all_history:
                                    all_history[new_username] = all_history.pop(username)
                                    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                                        json.dump(all_history, f, ensure_ascii=False, indent=2)
                            except Exception:
                                pass  # Pokud migrace selže, pokračujeme dál
                        
                        # Aktualizace config
                        usernames[new_username] = usernames.pop(username)
                        config["credentials"]["usernames"] = usernames
                        save_config(config)
                        
                        st.session_state["username"] = new_username
                        st.success("Uživatelské jméno bylo úspěšně změněno!")
                        st.rerun()
                
                st.divider()
                
                # Změna hesla
                st.subheader("Změna hesla")
                with st.form("change_password_form"):
                    old_password = st.text_input("Současné heslo", type="password")
                    new_password = st.text_input("Nové heslo (min. 4 znaky)", type="password")
                    confirm_new_password = st.text_input("Potvrzení nového hesla", type="password")
                    change_password_submitted = st.form_submit_button("Změnit heslo", use_container_width=True)
                
                if change_password_submitted:
                    if not old_password or not new_password or not confirm_new_password:
                        st.error("Vyplňte prosím všechna pole.")
                    elif len(new_password) < 4:
                        st.error("Nové heslo musí mít alespoň 4 znaky.")
                    elif new_password != confirm_new_password:
                        st.error("Nová hesla se neshodují.")
                    else:
                        # Ověření starého hesla pomocí bcrypt
                        try:
                            import bcrypt
                            stored_hash = usernames[username]["password"]
                            if bcrypt.checkpw(old_password.encode('utf-8'), stored_hash.encode('utf-8')):
                                # Heslo je správné, uložíme nové pomocí bcrypt
                                hashed_new_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                                
                                usernames[username]["password"] = hashed_new_password
                                config["credentials"]["usernames"] = usernames
                                save_config(config)
                                st.success("Heslo bylo úspěšně změněno!")
                                st.rerun()
                            else:
                                st.error("Současné heslo je nesprávné.")
                        except Exception as err:
                            st.error(f"Chyba při ověřování hesla: {err}")
        
        user_key = st.session_state.get("api_key_input", "") or stored_api_key
        ai_provider = st.session_state.get("ai_provider", stored_provider)
        api_key, _using_global_key = _resolve_api_key(ai_provider, user_key)
    
    st.title("🧬 RAD-seq Asistent (Lokalizované vyhledávání)")
    if st.session_state.get("conversation_title"):
        st.caption(f"📝 {st.session_state['conversation_title']}")
    
    # Dynamický info text podle poskytovatele
    provider_text = "Gemini" if ai_provider == "Gemini 2.0 Flash" else "GPT-4o (GitHub)"
    st.info(f"Vyhledávání v manuálech probíhá lokálně (bez limitů). {provider_text} se používá pouze pro generování textu.")
    
    if not st.session_state["messages"]:
        try:
            remote_history = get_chat_history(username)
        except Exception:
            remote_history = []
        if isinstance(remote_history, list) and remote_history:
            normalized = []
            for item in remote_history:
                if isinstance(item, dict):
                    content = item.get("content") or item.get("message", "")
                    role = item.get("role", "user")
                    normalized.append({"role": role, "content": content})
            if normalized:
                st.session_state["messages"] = normalized

    for msg in st.session_state["messages"]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(_format_citations(content))
    
    # Vstupní pole pro novou otázku
    if prompt := st.chat_input("Zadejte svůj dotaz (např. filtrování VCF nebo demultiplexing):"):
        user_key = st.session_state.get("api_key_input", "") or stored_api_key
        effective_key, from_secrets = _resolve_api_key(ai_provider, user_key)
        if ai_provider == "GPT-4o (GitHub)":
            if not effective_key:
                st.warning("⚠️ Pro GPT-4o (GitHub) nastavte GITHUB_TOKEN v .streamlit/secrets.toml.")
                st.stop()
        elif not effective_key:
            provider_name = "Google API klíč" if ai_provider == "Gemini 2.0 Flash" else "OpenAI API klíč"
            st.warning(f"⚠️ Prosím, zadejte {provider_name} v sekci 'Nastavení' v sidebaru, nebo ho uložte do svého profilu.")
            st.stop()
        api_key = effective_key
        
        st.session_state["messages"].append({"role": "user", "content": prompt})
        try:
            save_chat_to_db(username, prompt, "user")
        except Exception:
            pass
        with st.chat_message("user"):
            st.write(prompt)
        
        is_new_conversation = st.session_state["current_conversation_id"] is None
        if is_new_conversation:
            st.session_state["current_conversation_id"] = secrets.token_hex(4)
            try:
                st.session_state["conversation_title"] = generate_conversation_title(prompt, api_key, ai_provider)
            except Exception:
                st.session_state["conversation_title"] = f"Konverzace {datetime.now().strftime('%d.%m')}"
        
        with st.chat_message("assistant"):
            try:
                intent = classify_user_intent(prompt, api_key, ai_provider)
                gemini_model = None
                llm = None
                if ai_provider == "Gemini 2.0 Flash":
                    gemini_model = get_gemini_model(api_key)
                    if gemini_model is None:
                        st.error("Chyba při inicializaci Gemini modelu. Zkontrolujte prosím API klíč.")
                        st.session_state["messages"].pop()
                        st.stop()
                elif ai_provider != "GPT-4o (GitHub)":
                    llm = create_llm_instance(ai_provider, api_key)
                    if llm is None:
                        st.error("Chyba při inicializaci modelu. Zkontrolujte prosím API klíč.")
                        st.session_state["messages"].pop()
                        st.stop()
                relevant_docs = []
                general_knowledge_fallback = False
                context_text = ""

                if intent == "greeting":
                    full_prompt = PROMPT_TEMPLATE_NO_CONTEXT.format(question=prompt)
                else:
                    with st.spinner("Hledám v manuálech a připravuji odpověď..."):
                        vs = None
                        try:
                            vs = get_vectorstore_for_query(api_key, ai_provider)
                        except Exception:
                            vs = None
                            general_knowledge_fallback = True
                        if vs is None:
                            reason = _get_rag_failure_reason()
                            if reason:
                                st.warning(f"⚠️ Vyhledávání v manuálu není k dispozici: {reason}")
                            if general_knowledge_fallback:
                                full_prompt = PROMPT_TEMPLATE_NO_CONTEXT.format(question=prompt)
                            else:
                                full_prompt = PROMPT_TEMPLATE_NO_CONTEXT.format(question=prompt)
                        else:
                            try:
                                retriever = vs.as_retriever(search_kwargs={"k": 5})
                                relevant_docs = retriever.invoke(prompt)
                            except Exception:
                                relevant_docs = []
                                general_knowledge_fallback = True
                            context_text = ""
                            if relevant_docs and len(relevant_docs) > 0:
                                try:
                                    context_text = "\n\n".join([f"Zdroj: {d.metadata.get('source', 'Neznámý zdroj')}\n{d.page_content}" for d in relevant_docs])
                                except (IndexError, TypeError, AttributeError):
                                    context_text = ""
                            if context_text is None or not str(context_text).strip():
                                general_knowledge_fallback = True
                            if general_knowledge_fallback or not context_text or not str(context_text).strip():
                                full_prompt = PROMPT_TEMPLATE_NO_CONTEXT.format(question=prompt)
                            else:
                                full_prompt = PROMPT_TEMPLATE.format(context=context_text, question=prompt)

                if intent != "greeting" and len(context_text) == 0:
                    st.session_state["_last_context_length"] = 0
                    response_content = (
                        "**Znalostní báze není k dispozici.** Bez vyhledávacího indexu nemůžu odpovídat z manuálu.\n\n"
                        "**Lokálně:** V terminálu spusť `pip install sentence-transformers` a `python build_faiss_from_static.py`, pak obnov stránku.\n\n"
                        "**Na hostingu:** Do repozitáře nahraj složku `faiss_index_static` (soubory `index.faiss` a `index.pkl`) vygenerovanou lokálně; na některých hostinzích může být potřeba index sestavit na stejné platformě (Linux)."
                    )
                    streamed = False
                else:
                    st.session_state["_last_context_length"] = len(context_text or "")
                    with st.spinner("Hledám v manuálech a připravuji odpověď..." if intent != "greeting" else "Přemýšlím..."):
                        if ai_provider == "GPT-4o (GitHub)":
                            try:
                                response_content = get_github_model_response(prompt, context_text if context_text else "")
                                streamed = False
                            except Exception as e:
                                st.error(f"Chyba GitHub API: {e}")
                                response_content = ""
                                streamed = False
                        elif ai_provider == "Gemini 2.0 Flash" and gemini_model is not None:
                            response_content = get_gemini_response(gemini_model, full_prompt)
                            streamed = False
                        elif llm is not None:
                            response_content, streamed = _stream_or_invoke(llm, full_prompt)
                        else:
                            response_content = ""
                            streamed = False
                if not response_content:
                    st.error("Model nevrátil žádnou odpověď. Zkuste to prosím znovu.")
                    st.session_state["messages"].pop()
                    st.stop()
                _ctx_len = st.session_state.get("_last_context_length", 0)
                if not _ctx_len:
                    response_content = response_content.replace("[MANUAL]", "[AI]")
                if not streamed:
                    st.markdown(_format_citations(response_content))
                if _ctx_len and _ctx_len > 0:
                    st.info("📚 Odpověď sestavena na základě manuálu speciationgenomics.github.io")
                else:
                    st.warning("🤖 Odpověď generována z obecných znalostí modelu (v manuálu nenalezeno)")
                st.session_state["messages"].append({"role": "assistant", "content": response_content})
                try:
                    save_chat_to_db(username, response_content, "assistant")
                except Exception:
                    pass
                timestamp = datetime.now().isoformat()
                save_history(
                    username,
                    st.session_state["current_conversation_id"],
                    st.session_state["conversation_title"],
                    st.session_state["messages"],
                    timestamp
                )
                
                # Zobrazíme zdroje pouze pokud byly použity (technický dotaz s RAG)
                if intent == "technical" and relevant_docs and len(relevant_docs) > 0:
                    with st.expander("📚 Zobrazit použité zdroje z manuálu"):
                        for d in relevant_docs:
                            source = d.metadata.get('source', 'Neznámý zdroj')
                            st.write(f"- {source}")

            except Exception as e:
                st.error(f"Chyba modelu: {type(e).__name__}: {e}")
                if st.session_state["messages"]:  # Zkontrolujeme, zda existuje zpráva k odstranění
                    st.session_state["messages"].pop()  # Odstraníme chybnou zprávu

if __name__ == "__main__":
    main()