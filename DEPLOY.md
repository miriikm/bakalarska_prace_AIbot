# Nasazení na web (Streamlit Cloud) a Google přihlášení

## Chyba 403 od Google

Pokud po kliknutí na „Sign in with Google“ uvidíte **403. That’s an error. We're sorry, but you do not have access to this page**, jde téměř vždy o jednu z těchto příčin.

### 1. OAuth consent screen musí být „External“ (ne Internal)

Pokud je typ **Internal**, smí se přihlásit jen uživatelé z vaší Google Workspace organizace. Osobní Gmail (i jako test user) dostane 403.

**Postup:**

1. [Google Cloud Console](https://console.cloud.google.com/) → **APIs & Services** → **OAuth consent screen**.
2. Zkontrolujte **User type**: musí být **External**. Pokud je Internal, změňte na External (nebo vytvořte nový OAuth klient v projektu s External consent screen).
3. Uložte.

### 2. OAuth consent screen – Test users (režim Testing)

V režimu **Testing** smí přihlášení jen účty přidané jako **Test users**.

1. Na stránce **OAuth consent screen** sjeďte k **Test users**.
2. Klikněte **+ ADD USERS** a přidejte přesně ten e‑mail (Google účet), kterým se na webu přihlašujete.
3. Uložte.

### 3. Authorized JavaScript origins (důležité pro web)

Bez přidaného původu (origin) může Google u webové aplikace vracet 403.

**V Google Cloud Console:**

1. **APIs & Services** → **Credentials** → váš **OAuth 2.0 Client ID** (typ **Web application**).
2. V **Authorized JavaScript origins** klikněte **+ ADD URI** a přidejte:
   - `https://vase-app.streamlit.app` (bez lomítka na konci, vaše skutečná URL)
   - Např. `https://ai-pro-biology.streamlit.app`
3. Uložte.

### 4. Authorized redirect URIs

Redirect URI musí být **shodný** v aplikaci i v GCP.

**V tomtéž OAuth 2.0 Client ID:**

1. V **Authorized redirect URIs** přidejte přesně stejnou URL jako v Secrets (`redirect_uri`):
   - buď `https://vase-app.streamlit.app` (bez lomítka),
   - nebo `https://vase-app.streamlit.app/` (s lomítkem) – musí být **stejně** jako v Secrets.
2. Uložte změny.

**V nastavení aplikace na Streamlit Cloud (Secrets):**

Do Secrets (obsah jako v `.streamlit/secrets.toml`) přidejte v sekci `[firebase]` položku `redirect_uri` s toutéž URL:

```toml
[firebase]
client_id = "váš-client-id.apps.googleusercontent.com"
client_secret = "váš-client-secret"
redirect_uri = "https://VASE-APLIKACE.streamlit.app/"
api_key = "..."
auth_domain = "..."
project_id = "..."
storage_bucket = "..."
messaging_sender_id = "..."
app_id = "..."
```

- URL musí končit lomítkem `/` jen pokud tak máte v Google Console (obě místa musí být stejná).
- Žádné překlepy, žádné `http://` místo `https://`.

Po úpravě redirect URI v GCP i v Secrets aplikaci na Streamlit Cloud znovu nasaďte (nebo restartujte), aby se načetly nové Secrets.

### 5. Shrnutí – checklist při 403 na webu (localhost funguje)

| Kontrola | Kde |
|----------|-----|
| **User type = External** | OAuth consent screen (ne Internal) |
| **Váš e‑mail v Test users** | OAuth consent screen → Test users |
| **Authorized JavaScript origins** obsahuje `https://vase-app.streamlit.app` | Credentials → OAuth 2.0 Client ID (Web) |
| **Authorized redirect URIs** obsahuje stejnou URL jako v Secrets | Credentials → OAuth 2.0 Client ID |
| **Secrets na Streamlit Cloud** – `[firebase]` s `redirect_uri` = URL aplikace | Nastavení aplikace → Secrets |

Po úpravách v GCP chvíli počkejte (1–2 minuty) a zkuste přihlášení znovu; v anonymním okně vyzkoušejte, že se používá správný účet.
