# 🧬 RAD-seq Bioinfo Helper

## Popis

RAD-seq Bioinfo Helper je inteligentní asistent pro bioinformatiky pracující s RAD-seq daty. Aplikace využívá metodu **RAG (Retrieval-Augmented Generation)** k prohledávání odborných manuálů a podporuje **více AI modelů** (Google Gemini nebo OpenAI ChatGPT) pro generování Bash skriptů a odpovědí na otázky týkající se analýzy RAD-seq dat.

Aplikace kombinuje lokální vyhledávání v dokumentaci s pokročilými jazykovými modely, aby poskytovala přesné a praktické odpovědi založené na ověřených metodikách z odborných zdrojů.

## Hlavní funkce

### 🔐 Autentizace uživatelů
- **Registrace**: Vytvoření nového účtu s unikátním uživatelským jménem a e-mailem
- **Přihlášení**: Bezpečné přihlášení pomocí streamlit-authenticator
- **Správa profilu**: Možnost změny uživatelského jména a hesla
- **Perzistentní nastavení**: Bezpečné uložení vlastního Google API klíče do profilu

### 💬 Inteligentní chat s historií
- **Konverzační rozhraní**: Přirozená komunikace s asistentem pomocí chat rozhraní
- **Automatické titulky**: AI generuje krátké názvy (3-4 slova) pro každou konverzaci
- **Historie konverzací**: Ukládání a načítání předchozích konverzací podle uživatele
- **Nová konverzace**: Možnost začít novou konverzaci kdykoli

### 🔍 RAG systém (Retrieval-Augmented Generation)
- **Lokální dokumenty**: Vyhledávání v dokumentech uložených ve složce `docs/`
- **Online manuály**: Automatické načítání a indexace obsahu z webových stránek (např. speciationgenomics.github.io)
- **Vektorové vyhledávání**: Použití FAISS pro rychlé a přesné vyhledávání relevantních pasáží
- **Lokální embeddingy**: Použití HuggingFace embeddings (all-MiniLM-L6-v2) pro bezplatné a neomezené vyhledávání

### ⚙️ Perzistentní nastavení
- **Výběr AI modelu**: Možnost volby mezi Google Gemini a OpenAI ChatGPT
- **Uložení API klíče**: API klíč pro vybraný model lze uložit do profilu pro automatické použití
- **Bezpečné ukládání**: Citlivá data jsou uložena lokálně v `config.yaml` (není součástí repozitáře)

## Technologie

- **Streamlit** - Webové uživatelské rozhraní
- **LangChain** - Framework pro RAG a práci s jazykovými modely
- **FAISS** - Vektorová databáze pro efektivní vyhledávání podobných dokumentů
- **Google Gemini** - Jazykový model pro generování odpovědí a Bash skriptů (model: gemini-1.5-flash)
- **OpenAI ChatGPT** - Alternativní jazykový model (model: gpt-4o-mini)
- **HuggingFace Embeddings** - Lokální embeddingy pro vyhledávání (all-MiniLM-L6-v2)
- **streamlit-authenticator** - Autentizace a správa uživatelů
- **bcrypt** - Hashování hesel
- **pypdf** - Zpracování PDF dokumentů

## Instalace

### 1. Vytvoření virtuálního prostředí

```bash
python -m venv venv
```

**Aktivace na Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Aktivace na Linux/macOS:**
```bash
source venv/bin/activate
```

> **Poznámka pro Windows:** Pokud se zobrazí chyba "UnauthorizedAccess" při aktivaci, spusťte:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### 2. Instalace závislostí

```bash
pip install -r requirements.txt
```

**Seznam hlavních knihoven:**
- `streamlit>=1.31,<2.0` - Web framework
- `langchain>=1.2.0,<2.0.0` - RAG framework
- `langchain-community>=0.4.0,<1.0.0` - Komunitní integrace
- `langchain-openai>=1.1.0,<2.0.0` - OpenAI integrace
- `langchain-google-genai>=4.0.0,<5.0.0` - Google Gemini integrace
- `langchain-huggingface>=0.1.0,<1.0.0` - Lokální embeddingy
- `langchain-text-splitters>=1.1.0,<2.0.0` - Dělení textu
- `faiss-cpu>=1.7.4` - Vektorová databáze
- `streamlit-authenticator>=0.4.1,<0.5.0` - Autentizace
- `PyYAML>=6.0.0,<7.0.0` - Práce s YAML soubory
- `beautifulsoup4>=4.12.0` - Parsování HTML
- `requests>=2.31.0` - HTTP požadavky
- `pypdf>=3.0.0,<5.0.0` - Zpracování PDF souborů

### 3. Spuštění aplikace

```bash
streamlit run app.py
```

Aplikace se spustí na adrese `http://localhost:8501` (nebo jiný port, pokud je 8501 obsazen).

## Získání API klíčů

Aplikace podporuje dva AI modely. Pro použití aplikace potřebujete API klíč pro alespoň jeden z nich:

### 🤖 Google Gemini

1. **Navštivte** [Google AI Studio](https://aistudio.google.com/app/apikey)
2. **Přihlaste se** pomocí svého Google účtu
3. **Vytvořte nový API klíč** kliknutím na "Create API Key"
4. **Zkopírujte klíč** a uložte ho na bezpečné místo

> **💡 Tip**: Model **Gemini 1.5 Flash** je k dispozici **zdarma** s generosním denním limitem, což je ideální pro testování a osobní použití.

### 💬 OpenAI (ChatGPT)

1. **Navštivte** [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Přihlaste se** nebo vytvořte nový účet
3. **Přejděte** do sekce "API keys"
4. **Vytvořte nový klíč** kliknutím na "Create new secret key"
5. **Zkopírujte klíč** (začíná na `sk-`) a uložte ho na bezpečné místo

> **⚠️ Poznámka**: Pro použití OpenAI modelu je **vyžadován kredit** na vašem účtu. Model **gpt-4o-mini** je cenově výhodný a vhodný pro běžné použití.

## Konfigurace modelu

Po prvním přihlášení do aplikace si můžete nastavit preferovaný AI model:

1. **Otevřete sidebar** a rozbalte sekci **"⚙️ Nastavení a Profil"**
2. **Vyberte poskytovatele** z rozbalovacího seznamu:
   - **Google Gemini** - pro bezplatné použití
   - **OpenAI (ChatGPT)** - vyžaduje kredit
3. **Vložte API klíč** do textového pole (pole se automaticky přizpůsobí podle výběru)
4. **Klikněte na tlačítko** **"💾 Uložit API klíč do profilu"**
5. **Hotovo!** Aplikace si zapamatuje váš výběr a klíč bude automaticky použit při každém přihlášení

> **💡 Tip**: Můžete kdykoli přepnout mezi modely nebo změnit API klíč. Aktivní model je vždy zobrazen v horní části sidebaru.

### API klíče z .env (volitelné)

API klíče lze nastavit i přes soubor **`.env`** v kořeni projektu (není v gitu). Zkopírujte `env.example` na `.env` a doplňte hodnoty:

- `GOOGLE_API_KEY` nebo `GEMINI_API_KEY` – pro Google Gemini
- `OPENAI_API_KEY` – pro OpenAI (začíná na `sk-`)

Priorita: klíč uložený v Nastavení (profil) → .env → Streamlit secrets / proměnné prostředí.

## Použití

1. **Registrace/ Přihlášení**: Při prvním spuštění se zaregistrujte nebo přihlaste
2. **Konfigurace modelu**: 
   - Postupujte podle sekce [**Konfigurace modelu**](#konfigurace-modelu) výše
   - Vyberte preferovaný AI model a uložte API klíč
3. **Nahrání dokumentů** (volitelné): 
   - V sidebaru rozbalte sekci **"📚 Správa manuálů"**
   - Nahrajte `.md`, `.txt` nebo `.pdf` soubory pomocí file uploaderu
   - Klikněte na **"Rebuild Knowledge Base"** pro integraci nových dokumentů
4. **Položení otázky**: Zadejte dotaz týkající se RAD-seq analýzy do chat rozhraní
5. **Získání odpovědi**: Aplikace vyhledá relevantní informace a vygeneruje odpověď s Bash příkazy

## Nasazení na hosting (Streamlit Cloud apod.)

Aby na hostingu fungovalo vyhledávání v manuálu (RAG), musí být v repozitáři složka **`faiss_index_static/`** včetně souborů **`index.faiss`** a **`index.pkl`** (vygenerované příkazem `python build_faiss_from_static.py`). Pokud index na hostingu nejde načíst (např. byl sestaven na Windows a hosting běží na Linuxu), aplikace zobrazí varování s důvodem a odpověď bude bez kontextu z manuálu. V takovém případě sestavte index ve stejném prostředí jako hosting (např. lokálně na Linuxu nebo v CI) a nahrajte ho do gitu.

## Bezpečnostní poznámka

⚠️ **DŮLEŽITÉ**: Citlivá data (API klíče pro Google Gemini nebo OpenAI, hesla) jsou uložena lokálně v souboru `config.yaml` a **nejsou součástí repozitáře**. Tento soubor je automaticky ignorován pomocí `.gitignore`.

**Ignorované soubory:**
- **`config.yaml`** - Uživatelské účty, API klíče (Google/OpenAI) a volba AI modelu
- **`.env`** - API klíče načtené z prostředí (GOOGLE_API_KEY, OPENAI_API_KEY atd.)
- **`history.json`** - Historie konverzací všech uživatelů
- **`venv/`** - Virtuální prostředí Pythonu
- **`docs/`** - Nahrané dokumenty a manuály
- **`faiss_index_local/`** - Vektorový index pro vyhledávání
- **`.streamlit/`** - Streamlit konfigurace

**Nikdy necommitujte tyto soubory do repozitáře!** Chráníte tak nejen své API klíče, ale i osobní data a historii konverzací.

## Struktura projektu

```
BP/
├── app.py                 # Hlavní aplikace
├── requirements.txt       # Seznam závislostí
├── README.md             # Tento soubor
├── .gitignore            # Ignorované soubory
├── pyrightconfig.json    # Konfigurace type checkeru
├── config.yaml           # ⚠️ Citlivá data (lokálně, není v repozitáři)
├── history.json          # ⚠️ Historie konverzací (lokálně, není v repozitáři)
├── docs/                 # ⚠️ Nahrané dokumenty (lokálně, není v repozitáři)
├── faiss_index_local/    # ⚠️ Vektorový index (lokálně, není v repozitáři)
└── venv/                 # ⚠️ Virtuální prostředí (lokálně, není v repozitáři)
```

### 📁 Důležité soubory a složky

- **`docs/`** - Složka pro vlastní manuály a dokumenty. Můžete sem nahrát:
  - **Markdown soubory** (`.md`) - například dokumentace nástrojů
  - **Textové soubory** (`.txt`) - jednoduché textové manuály
  - **PDF soubory** (`.pdf`) - často používané formáty v bioinformatice
  - Po nahrání souborů klikněte na **"Rebuild Knowledge Base"** v sidebaru pro jejich integraci do vyhledávání

- **`history.json`** - Automaticky vytvářený soubor obsahující historii všech konverzací. Každý uživatel má svou vlastní historii, která je strukturovaná podle uživatelského jména. Historie se automaticky ukládá po každé odpovědi asistenta a obsahuje:
  - Unikátní ID konverzace
  - Název konverzace (vygenerovaný AI)
  - Časové razítko
  - Kompletní seznam zpráv (otázky a odpovědi)

## Autor

[Vaše jméno] - Bakalářská práce

---

**Licence**: [Doplňte licenci, pokud je potřeba]
