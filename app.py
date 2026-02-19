import streamlit as st
import os
import shutil
import json
import yaml
import secrets
from datetime import datetime
from pathlib import Path
from pydantic import SecretStr

import streamlit_authenticator as stauth

from langchain_community.document_loaders import RecursiveUrlLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
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
        loader = RecursiveUrlLoader(url=url, max_depth=2)
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

def create_llm_instance(provider: str, api_key: str):
    """Vytvoří instanci LLM podle vybraného poskytovatele."""
    if not api_key or api_key is None or (isinstance(api_key, str) and api_key.strip() == ""):
        return None
    
    try:
        if provider == "Google Gemini":
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        elif provider == "OpenAI (ChatGPT)":
            # Pro OpenAI použijeme SecretStr pro správné typování
            return ChatOpenAI(model="gpt-4o-mini", api_key=SecretStr(api_key), temperature=0.7)
        else:
            return None
    except Exception as e:
        st.error(f"Chyba při inicializaci {provider}: {e}")
        return None

def generate_conversation_title(first_question: str, api_key: str, provider: str = "Google Gemini") -> str:
    """Vygeneruje krátký název (3-4 slova) pro konverzaci pomocí vybraného LLM."""
    try:
        # Validace API klíče
        if not api_key or api_key is None or (isinstance(api_key, str) and api_key.strip() == ""):
            return f"Konverzace {datetime.now().strftime('%d.%m')}"
        
        # Vytvoření LLM instance
        llm = create_llm_instance(provider, api_key)
        if llm is None:
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
Jsi přátelský a vysoce odborný asistent pro RAD-seq analýzy.

Pokud je dotaz odborný, použij poskytnutý KONTEXT z manuálů. Pokud v KONTEXTU odpověď není, využij své všeobecné znalosti, ale upozorni uživatele, že tato informace pochází z tvých obecných znalostí a ne z nahraných manuálů.

**Formátování odpovědi (vždy používej Markdown):**
- **Důležité termíny a koncepty** označuj tučně (např. **RAD-seq**, **VCF soubor**)
- Bash příkazy vždy v blocích kódu:
  ```bash
  příkaz --parametr hodnota
  ```
- Používej odrážky (-) a číslované seznamy (1., 2., ...) pro přehlednost
- Strukturované sekce s nadpisy (## nebo ###)
- Pokud generuješ více příkazů, seskupte je logicky do sekcí

**Příklad struktury odpovědi:**
## Název řešení

**Klíčový koncept:** Vysvětlení...

### Postup:
1. První krok
2. Druhý krok

### Bash příkazy:
```bash
příkaz1
příkaz2
```

KONTEXT z manuálů:
{context}

Dotaz uživatele:
{question}
"""

PROMPT_TEMPLATE_NO_CONTEXT = """
Jsi přátelský a vysoce odborný asistent pro RAD-seq analýzy.

Pokud ti uživatel posílá běžný pozdrav nebo neformální zprávu, odpověz mu lidsky a přátelsky. Nabídni pomoc s bioinformatikou a RAD-seq analýzami.

Pokud je dotaz technický, ale nemám k dispozici kontext z manuálů, odpověz na základě svých obecných znalostí, ale upozorni uživatele na to.

**Formátování odpovědi (vždy používej Markdown):**
- **Důležité termíny** označuj tučně
- Bash příkazy v blocích kódu: ```bash ... ```
- Používej odrážky a číslované seznamy pro přehlednost
- Strukturované sekce s nadpisy

Dotaz uživatele:
{question}
"""

def classify_user_intent(prompt: str, api_key: str, provider: str = "Google Gemini") -> str:
    """Rozhodne, zda je dotaz pozdrav/neformální nebo technický dotaz vyžadující RAG."""
    if not api_key or api_key is None or (isinstance(api_key, str) and api_key.strip() == ""):
        return "technical"  # Default na technický, pokud není API klíč
    
    # Jednoduchá kontrola klíčových slov pro pozdravy
    greeting_keywords = ["ahoj", "čau", "dobrý den", "dobrý večer", "děkuji", "děkuju", "díky", 
                         "co umíš", "co dokážeš", "pomoc", "help", "hello", "hi", "thanks", "thank you"]
    prompt_lower = str(prompt).lower().strip()
    
    # Pokud je dotaz velmi krátký a obsahuje pozdrav, pravděpodobně jde o pozdrav
    if len(prompt_lower.split()) <= 5:
        for keyword in greeting_keywords:
            if keyword in prompt_lower:
                return "greeting"
    
    # Pokud začíná pozdravem, ale pokračuje technickým dotazem, použijeme RAG
    # Použijeme LLM pro přesnější klasifikaci
    try:
        llm = create_llm_instance(provider, api_key)
        if llm is None:
            return "technical"
        
        classification_prompt = f"""Rozhodni, zda je tento dotaz:
1. POZDRAV/NEformální zpráva (např. "Ahoj", "Děkuji", "Co umíš?") - odpověz "greeting"
2. TECHNICKÝ dotaz týkající se bioinformatiky/RAD-seq (např. "Jak filtrovat VCF?", "Demultiplexing") - odpověz "technical"

Dotaz: "{prompt}"

Odpověz pouze jedním slovem: "greeting" nebo "technical":"""
        
        response = llm.invoke(classification_prompt)
        if response and hasattr(response, 'content') and response.content:
            result = str(response.content).strip().lower()
            if "greeting" in result:
                return "greeting"
    except Exception:
        pass  # Pokud selže klasifikace, použijeme RAG jako default
    
    return "technical"

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
        # Načtení .md, .txt a .pdf souborů
        for pattern in ["**/*.md", "**/*.txt", "**/*.pdf"]:
            try:
                local_loader = DirectoryLoader(str(DOCS_DIR), glob=pattern)
                all_docs.extend(local_loader.load())
            except Exception as e:
                # Pokud některý loader selže (např. chybí pypdf pro PDF), pokračujeme dál
                continue

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
    
    # Načtení API klíče a poskytovatele z profilu uživatele
    usernames = config.get("credentials", {}).get("usernames", {})
    stored_api_key = usernames.get(username, {}).get("api_key", "") if username in usernames else ""
    stored_provider = usernames.get(username, {}).get("ai_provider", "Google Gemini") if username in usernames else "Google Gemini"
    
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
        provider_icon = "🤖" if current_provider == "Google Gemini" else "💬"
        st.caption(f"{provider_icon} **Aktivní model:** {current_provider}")
        
        st.divider()
        
        # Tlačítko "Nová konverzace"
        if st.button("➕ Nová konverzace", use_container_width=True, type="primary"):
            st.session_state["messages"] = []
            st.session_state["current_conversation_id"] = None
            st.session_state["conversation_title"] = None
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
        
        # SEKCE MANUÁLY - Expander se správou manuálů
        with st.expander("📚 Správa manuálů", expanded=False):
            # Nahrání souboru
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
                                    if os.path.exists("faiss_index_local"):
                                        shutil.rmtree("faiss_index_local")
                                    # Vymazání cache pro build_vectorstore
                                    st.cache_resource.clear()
                                    vs = build_vectorstore()
                                
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
                    if os.path.exists("faiss_index_local"):
                        shutil.rmtree("faiss_index_local")
                    # Vymazání cache pro build_vectorstore
                    st.cache_resource.clear()
                    vs = build_vectorstore()

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
            provider_options = ["Google Gemini", "OpenAI (ChatGPT)"]
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
            
            # Dynamický popisek podle vybraného poskytovatele
            api_key_label = "Google API Key" if selected_provider == "Google Gemini" else "OpenAI API Key"
            api_key_help = f"Zadejte nebo upravte svůj {api_key_label}. Pro trvalé uložení použijte tlačítko 'Uložit API klíč' níže."
            
            # API klíč - pole pro zadání/změnu
            if stored_api_key:
                st.info(f"✅ API klíč je uložen v profilu a bude automaticky použit pro {selected_provider}.")
            
            api_key_input = st.text_input(
                api_key_label, 
                value=st.session_state["api_key_input"],
                type="password",
                help=api_key_help,
                key="api_key_input_field"
            )
            
            # Uložíme do session_state
            if api_key_input:
                st.session_state["api_key_input"] = api_key_input
            
            # Kontrola, zda má uživatel správný klíč pro vybraný poskytovatele
            if selected_provider == "OpenAI (ChatGPT)" and (not api_key_input or not api_key_input.startswith("sk-")):
                if not stored_api_key or not stored_api_key.startswith("sk-"):
                    st.warning("⚠️ Pro použití OpenAI (ChatGPT) musíte zadat platný OpenAI API klíč (začíná na 'sk-').")
            
            # Uložení API klíče do profilu
            if api_key_input and api_key_input != stored_api_key:
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
        
        # Použijeme API klíč z inputu, nebo pokud je prázdný, použijeme uložený z profilu
        api_key = st.session_state.get("api_key_input", "") if st.session_state.get("api_key_input", "") else stored_api_key
        # Použijeme poskytovatele z session_state nebo uloženého
        ai_provider = st.session_state.get("ai_provider", stored_provider)
    
    # Hlavní chat rozhraní
    st.title("🧬 RAD-seq Asistent (Lokalizované vyhledávání)")
    if st.session_state.get("conversation_title"):
        st.caption(f"📝 {st.session_state['conversation_title']}")
    
    # Dynamický info text podle poskytovatele
    provider_text = "Gemini" if ai_provider == "Google Gemini" else "ChatGPT"
    st.info(f"Vyhledávání v manuálech probíhá lokálně (bez limitů). {provider_text} se používá pouze pro generování textu.")
    
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
            provider_name = "Google API klíč" if ai_provider == "Google Gemini" else "OpenAI API klíč"
            st.warning(f"⚠️ Prosím, zadejte {provider_name} v sekci 'Nastavení' v sidebaru, nebo ho uložte do svého profilu.")
            st.stop()
        
        # Kontrola, zda má uživatel správný klíč pro vybraný poskytovatele
        if ai_provider == "OpenAI (ChatGPT)" and not api_key.startswith("sk-"):
            st.warning("⚠️ Pro použití OpenAI (ChatGPT) musíte zadat platný OpenAI API klíč (začíná na 'sk-'). Zadejte ho v sekci 'Nastavení'.")
            st.stop()
        
        # Přidáme uživatelskou zprávu
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Pokud je to první zpráva v konverzaci, vytvoříme novou konverzaci a vygenerujeme název
        is_new_conversation = st.session_state["current_conversation_id"] is None
        if is_new_conversation:
            # Generování unikátního ID pomocí secrets.token_hex(4)
            st.session_state["current_conversation_id"] = secrets.token_hex(4)
            with st.spinner("Vytvářím název konverzace..."):
                try:
                    title = generate_conversation_title(prompt, api_key, ai_provider)
                    st.session_state["conversation_title"] = title
                except Exception as e:
                    st.session_state["conversation_title"] = f"Konverzace {datetime.now().strftime('%d.%m')}"
        
        # Generování odpovědi
        with st.chat_message("assistant"):
            try:
                # ROZLIŠENÍ ZÁMĚRU (Routing) - rozhodneme, zda použít RAG nebo odpovědět přímo
                intent = classify_user_intent(prompt, api_key, ai_provider)
                
                # Dynamická inicializace LLM modelu podle poskytovatele
                llm = create_llm_instance(ai_provider, api_key)
                if llm is None:
                    provider_name = "Gemini" if ai_provider == "Google Gemini" else "ChatGPT"
                    st.error(f"Chyba při inicializaci {provider_name} modelu. Zkontrolujte prosím API klíč.")
                    st.session_state["messages"].pop()  # Odstraníme chybnou zprávu
                    st.stop()
                
                # Inicializace relevant_docs pro případ, že nebude použito RAG
                relevant_docs = []
                
                # Pokud jde o pozdrav/neformální zprávu, odpovíme přímo bez RAG
                if intent == "greeting":
                    with st.spinner("Přemýšlím..."):
                        full_prompt = PROMPT_TEMPLATE_NO_CONTEXT.format(question=prompt)
                        response = llm.invoke(full_prompt)
                else:
                    # Technický dotaz - použijeme RAG
                    with st.spinner("Prohledávám manuály na pozadí..."):
                        vs = build_vectorstore()
                        if vs is None:
                            # Pokud není vektorová databáze, použijeme LLM bez kontextu
                            st.warning("Znalostní báze není k dispozici. Odpovídám na základě obecných znalostí.")
                            full_prompt = PROMPT_TEMPLATE_NO_CONTEXT.format(question=prompt)
                            response = llm.invoke(full_prompt)
                            relevant_docs = []
                        else:
                            # Najdeme 5 nejrelevantnějších pasáží v manuálu
                            retriever = vs.as_retriever(search_kwargs={"k": 5})
                            relevant_docs = retriever.invoke(prompt)
                            
                            # OŠETŘENÍ PRÁZDNÉHO KONTEXTU
                            if relevant_docs and len(relevant_docs) > 0:
                                context_text = "\n\n".join([f"Zdroj: {d.metadata.get('source', 'Neznámý zdroj')}\n{d.page_content}" for d in relevant_docs])
                                full_prompt = PROMPT_TEMPLATE.format(context=context_text, question=prompt)
                            else:
                                # Pokud není kontext, použijeme prompt bez kontextu
                                st.info("V manuálech nebyla nalezena relevantní informace. Odpovídám na základě obecných znalostí.")
                                full_prompt = PROMPT_TEMPLATE_NO_CONTEXT.format(question=prompt)
                                context_text = ""
                            
                            response = llm.invoke(full_prompt)

                # Ošetření odpovědi - zkontrolujeme, zda response a response.content existují
                if response and hasattr(response, 'content') and response.content:
                    response_content = str(response.content) if response.content else ""
                else:
                    st.error("Model nevrátil žádnou odpověď. Zkuste to prosím znovu.")
                    st.session_state["messages"].pop()  # Odstraníme chybnou zprávu
                    st.stop()

                # Zobrazíme odpověď (formátování je už v Markdownu z promptu)
                st.markdown(response_content)
                
                # Přidáme odpověď do zpráv
                st.session_state["messages"].append({"role": "assistant", "content": response_content})
                
                # AUTOMATICKÉ UKLÁDÁNÍ: Uložíme konverzaci do historie po každé odpovědi
                # CHYTRÁ AKTUALIZACE: Pokud existuje stejné ID, aktualizujeme existující záznam
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
                st.error(f"Došlo k chybě: {e}")
                if st.session_state["messages"]:  # Zkontrolujeme, zda existuje zpráva k odstranění
                    st.session_state["messages"].pop()  # Odstraníme chybnou zprávu

if __name__ == "__main__":
    main()