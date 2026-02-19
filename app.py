import streamlit as st
import os
import shutil
import json
import uuid
import yaml
import secrets
from datetime import datetime
from pathlib import Path

import streamlit_authenticator as stauth

from langchain_community.document_loaders import RecursiveUrlLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings # Lokální embedding bez limitů
from langchain_community.vectorstores import FAISS

# --- KONFIGURACE ---
BASE_URL = "https://speciationgenomics.github.io/"
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)
HISTORY_FILE = Path("history.json")
CONFIG_FILE = Path("config.yaml")

# --- SPRÁVA KONFIGURACE A AUTENTIZACE ---
def load_config() -> dict:
    """Načte konfiguraci z config.yaml."""
    if not CONFIG_FILE.exists():
        # Vytvoříme výchozí konfiguraci
        default_config = {
            "credentials": {"usernames": {}},
            "cookie": {
                "expiry_days": 0,  # Cookies expirují při zavření prohlížeče
                "key": secrets.token_hex(16),
                "name": "radseq_auth"
            },
            "preauthorized": {"emails": []}
        }
        save_config(default_config)
        return default_config
    
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        default_config = {
            "credentials": {"usernames": {}},
            "cookie": {
                "expiry_days": 0,  # Cookies expirují při zavření prohlížeče
                "key": secrets.token_hex(16),
                "name": "radseq_auth"
            },
            "preauthorized": {"emails": []}
        }
        save_config(default_config)
        return default_config

def save_config(config: dict):
    """Uloží konfiguraci do config.yaml."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

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
def load_history(user_email: str) -> dict:
    """Načte historii konverzací pro daného uživatele (podle emailu)."""
    if not user_email:
        return {}
    if not HISTORY_FILE.exists():
        return {}
    
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            all_history = json.load(f)
        email_key = str(user_email).lower() if user_email else ""
        return all_history.get(email_key, {}) if email_key else {}
    except Exception:
        return {}

def save_history(user_email: str, conversation_id: str, title: str, messages: list, timestamp: str):
    """Uloží konverzaci do historie (podle emailu)."""
    if not user_email:
        return
    all_history = {}
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                all_history = json.load(f)
        except Exception:
            pass
    
    email_key = str(user_email).lower() if user_email else ""
    if not email_key:
        return
    
    if email_key not in all_history:
        all_history[email_key] = {}
    
    all_history[email_key][conversation_id] = {
        "id": conversation_id,
        "timestamp": timestamp,
        "title": title,
        "messages": messages
    }
    
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_history, f, ensure_ascii=False, indent=2)

def migrate_history_username_to_email(old_username: str, new_email: str, config: dict):
    """Migruje historii z username na email při změně username."""
    if not new_email or not HISTORY_FILE.exists():
        return
    
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            all_history = json.load(f)
        
        # Pokud existuje historie pod starým username, přesuneme ji na email
        if old_username in all_history:
            email_key = str(new_email).lower() if new_email else ""
            if email_key:
                all_history[email_key] = all_history.pop(old_username)
                with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(all_history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

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
                st.success("Účet byl úspěšně vytvořen! Nyní se můžete přihlásit.")
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
                # Migrace historie
                migrate_history_username_to_email(username, current_email, config)
                
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

def generate_conversation_title(first_question: str, api_key: str) -> str:
    """Vygeneruje krátký název (3-4 slova) pro konverzaci pomocí Gemini."""
    try:
        # Validace API klíče
        if not api_key or api_key is None or (isinstance(api_key, str) and api_key.strip() == ""):
            return f"Konverzace {datetime.now().strftime('%d.%m')}"
        
        # Try-except pro inicializaci LLM modelu
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        except Exception as llm_error:
            return f"Konverzace {datetime.now().strftime('%d.%m')}"
        
        prompt = f"Vytvoř velmi krátký název (přesně 3-4 slova) pro tuto konverzaci založenou na této otázce: {first_question}\n\nNázev (pouze 3-4 slova, bez uvozovek):"
        response = llm.invoke(prompt)
        
        # Ošetření odpovědi - zkontrolujeme, zda response a response.content existují
        if response and hasattr(response, 'content') and response.content:
            content = str(response.content) if response.content else ""
        else:
            return f"Konverzace {datetime.now().strftime('%d.%m')}"
        
        if isinstance(content, list):
            content = str(content[0]) if content and len(content) > 0 else ""
        
        title = str(content).strip().strip('"').strip("'")
        # Omezíme na 4 slova
        words = title.split()[:4]
        return " ".join(words) if words else "Nová konverzace"
    except Exception as e:
        return f"Konverzace {datetime.now().strftime('%d.%m')}"

PROMPT_TEMPLATE = """
Jsi expert na bioinformatiku a sekvenování RAD-seq.
Na základě poskytnutého kontextu z manuálů vygeneruj odpověď pro uživatele.
Odpověď musí obsahovat konkrétní Bash příkazy a stručné vysvětlení.

Kontext z manuálů:
{context}

Dotaz uživatele:
{question}
"""

@st.cache_resource(show_spinner=True)
def build_vectorstore():
    # Používáme lokální model, který běží u vás v počítači (zdarma a bez API limitů)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    index_path = "faiss_index_local"
    
    # Pokud už index existuje na disku, prostě ho načteme
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # Pokud ne, stáhneme data a vytvoříme ho
    all_docs = []
    try:
        loader = RecursiveUrlLoader(url=BASE_URL, max_depth=2)
        all_docs.extend(loader.load())
    except:
        st.warning("Nepodařilo se načíst web, zkouším lokální soubory.")

    if any(DOCS_DIR.iterdir()):
        local_loader = DirectoryLoader(str(DOCS_DIR), glob="**/*.md")
        all_docs.extend(local_loader.load())

    if not all_docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def main():
    st.set_page_config(page_title="RAD-seq Bioinfo Helper", layout="wide")
    
    # Načtení konfigurace
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
    
    # Načtení API klíče z profilu uživatele
    usernames = config.get("credentials", {}).get("usernames", {})
    stored_api_key = usernames.get(username, {}).get("api_key", "") if username in usernames else ""
    
    # Inicializace session state pro zprávy a aktuální konverzaci
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "current_conversation_id" not in st.session_state:
        st.session_state["current_conversation_id"] = None
    if "conversation_title" not in st.session_state:
        st.session_state["conversation_title"] = None
    
    # Načtení historie pro uživatele (podle emailu)
    user_history = load_history(user_email)
    
    with st.sidebar:
        # Logout tlačítko pomocí authenticatoru
        try:
            authenticator.logout('Odhlásit se', 'sidebar')
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
        
        # Správa profilu
        show_profile_management(config, username)
        
        st.divider()
        
        st.header("📚 Historie konverzací")
        
        # Tlačítko "Nová konverzace"
        if st.button("➕ Nová konverzace", use_container_width=True, type="primary"):
            st.session_state["messages"] = []
            st.session_state["current_conversation_id"] = None
            st.session_state["conversation_title"] = None
            st.rerun()
        
        st.divider()
        
        # Zobrazení historie konverzací
        if user_history:
            st.subheader("Předchozí konverzace")
            # Seřadíme podle časového razítka (nejnovější první)
            sorted_conversations = sorted(
                user_history.items(),
                key=lambda x: x[1].get("timestamp", ""),
                reverse=True
            )
            
            for conv_id, conv_data in sorted_conversations:
                title = conv_data.get("title", "Bez názvu")
                timestamp = conv_data.get("timestamp", "")
                # Zobrazíme tlačítko s názvem a datem
                button_label = f"{title}\n📅 {timestamp[:10] if len(timestamp) >= 10 else timestamp}"
                if st.button(button_label, key=f"hist_{conv_id}", use_container_width=True):
                    # Načteme konverzaci do session_state
                    st.session_state["messages"] = conv_data.get("messages", [])
                    st.session_state["current_conversation_id"] = conv_id
                    st.session_state["conversation_title"] = title
                    st.rerun()
        else:
            st.info("Zatím žádné uložené konverzace.")
        
        st.divider()
        st.header("Nastavení")
        
        # Textové pole pro API klíč - předvyplníme, pokud je uložen v profilu
        if stored_api_key:
            st.info("✅ API klíč je uložen v profilu a bude automaticky použit.")
        
        api_key_input = st.text_input(
            "Google API Key", 
            value=stored_api_key if stored_api_key else "",
            type="password",
            help="Zadejte nebo upravte svůj Google API klíč. Pro trvalé uložení použijte sekci 'Můj profil'."
        )
        
        # Použijeme API klíč z inputu, nebo pokud je prázdný, použijeme uložený z profilu
        api_key = api_key_input if api_key_input else stored_api_key

        uploaded_file = st.file_uploader(
            "Nahraj manuál (.md nebo .txt)",
            type=["md", "txt"],
        )

        if uploaded_file is not None:
            save_path = DOCS_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Soubor '{uploaded_file.name}' byl uložen do složky 'docs/'.")

        if st.button("Rebuild Knowledge Base"):
            with st.spinner("Přestavuji znalostní bázi..."):
                if os.path.exists("faiss_index_local"):
                    shutil.rmtree("faiss_index_local")
                build_vectorstore.clear()
                vs = build_vectorstore()

            if vs is not None:
                st.success("Znalostní báze byla úspěšně znovu vytvořena a nový manuál byl integrován.")
            else:
                st.warning("Znalostní bázi se nepodařilo znovu vytvořit – nebyla nalezena žádná data k indexaci.")
    
    # Hlavní chat rozhraní
    st.title("🧬 RAD-seq Asistent (Lokalizované vyhledávání)")
    if st.session_state.get("conversation_title"):
        st.caption(f"📝 {st.session_state['conversation_title']}")
    st.info("Vyhledávání v manuálech probíhá lokálně (bez limitů). Gemini se používá pouze pro generování textu.")
    
    # Zobrazení historie zpráv
    for msg in st.session_state["messages"]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)
    
    # Vstupní pole pro novou otázku
    if prompt := st.chat_input("Zadejte svůj dotaz (např. filtrování VCF nebo demultiplexing):"):
        # Validace API klíče - zkontrolujeme, zda není None nebo prázdný string
        if not api_key or api_key is None or (isinstance(api_key, str) and api_key.strip() == ""):
            st.warning("⚠️ Prosím, zadejte Google API klíč v sekci 'Nastavení' v sidebaru, nebo ho uložte do svého profilu v sekci 'Můj profil'.")
            st.stop()
        
        # Přidáme uživatelskou zprávu
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Pokud je to první zpráva v konverzaci, vytvoříme novou konverzaci a vygenerujeme název
        if st.session_state["current_conversation_id"] is None:
            st.session_state["current_conversation_id"] = str(uuid.uuid4())
            with st.spinner("Vytvářím název konverzace..."):
                try:
                    title = generate_conversation_title(prompt, api_key)
                    st.session_state["conversation_title"] = title
                except Exception as e:
                    st.session_state["conversation_title"] = f"Konverzace {datetime.now().strftime('%d.%m')}"
        
        # Generování odpovědi
        with st.chat_message("assistant"):
            with st.spinner("Prohledávám manuály na pozadí..."):
                try:
                    vs = build_vectorstore()
                    if vs is None:
                        st.error("Nenalezena žádná data k analýze.")
                        st.stop()

                    # Najdeme 5 nejrelevantnějších pasáží v manuálu
                    retriever = vs.as_retriever(search_kwargs={"k": 5})
                    relevant_docs = retriever.invoke(prompt)
                    
                    context_text = "\n\n".join([f"Zdroj: {d.metadata.get('source')}\n{d.page_content}" for d in relevant_docs])

                    # Gemini použijeme jen na finální zpracování odpovědi
                    # Try-except pro inicializaci LLM modelu
                    try:
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
                    except Exception as llm_error:
                        st.error(f"Chyba při inicializaci Gemini modelu: {llm_error}")
                        st.session_state["messages"].pop()  # Odstraníme chybnou zprávu
                        st.stop()
                    
                    full_prompt = PROMPT_TEMPLATE.format(context=context_text, question=prompt)
                    response = llm.invoke(full_prompt)

                    # Ošetření odpovědi - zkontrolujeme, zda response a response.content existují
                    if response and hasattr(response, 'content') and response.content:
                        response_content = str(response.content) if response.content else ""
                    else:
                        st.error("Model nevrátil žádnou odpověď. Zkuste to prosím znovu.")
                        st.session_state["messages"].pop()  # Odstraníme chybnou zprávu
                        st.stop()

                    # Zobrazíme odpověď
                    st.markdown(response_content)
                    
                    # Přidáme odpověď do zpráv
                    st.session_state["messages"].append({"role": "assistant", "content": response_content})
                    
                    # Uložíme konverzaci do historie (podle emailu)
                    timestamp = datetime.now().isoformat()
                    save_history(
                        user_email,
                        st.session_state["current_conversation_id"],
                        st.session_state["conversation_title"],
                        st.session_state["messages"],
                        timestamp
                    )
                    
                    with st.expander("Zobrazit použité zdroje z manuálu"):
                        for d in relevant_docs:
                            st.write(f"- {d.metadata.get('source')}")

                except Exception as e:
                    st.error(f"Došlo k chybě: {e}")
                    if st.session_state["messages"]:  # Zkontrolujeme, zda existuje zpráva k odstranění
                        st.session_state["messages"].pop()  # Odstraníme chybnou zprávu

if __name__ == "__main__":
    main()