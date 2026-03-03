# Nasazení na web (Streamlit Cloud) a Google přihlášení

## Chyba 403 od Google

Pokud po kliknutí na „Sign in with Google“ uvidíte **403. That’s an error. We're sorry, but you do not have access to this page**, jde téměř vždy o jednu z těchto příčin.

### 1. OAuth consent screen je v režimu „Testing“

V režimu **Testing** smí přihlášení jen účty přidané jako **Test users**.

**Postup:**

1. Otevřete [Google Cloud Console](https://console.cloud.google.com/) → **APIs & Services** → **OAuth consent screen**.
2. V sekci **Test users** klikněte na **+ ADD USERS** a přidejte e‑mailové adresy, které smí přihlášení používat.
3. Uložte. Přihlášení z těchto účtů pak 403 nedostanou.

Alternativa: přepnout aplikaci do **Production** (pro veřejné použití může být potřeba ověření aplikace u Google).

### 2. Nesprávný nebo chybějící redirect URI

Redirect URI musí být **shodný** v aplikaci i v Google Cloud a musí odpovídat adrese, na které aplikace na webu běží.

**V Google Cloud Console:**

1. **APIs & Services** → **Credentials** → vyberte váš **OAuth 2.0 Client ID** (typ Web application).
2. V **Authorized redirect URIs** přidejte přesně tuto adresu (nahraďte za vaši skutečnou URL aplikace):
   - `https://VASE-APLIKACE.streamlit.app/`
   - Např. `https://bakalarska-prace-aibot.streamlit.app/`
3. Uložte změny.

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

### 3. Shrnutí

| Problém | Řešení |
|--------|--------|
| 403 po kliknutí na Google přihlášení | Přidat svůj e‑mail (nebo jiné uživatele) do **Test users** v OAuth consent screen, nebo přepnout na Production. |
| Redirect URI mismatch (jiná chyba od Google) | Do **Authorized redirect URIs** v GCP a do Secrets `[firebase]` `redirect_uri` zadat přesně stejnou URL aplikace (např. `https://xxx.streamlit.app/`). |

Po těchto úpravách by přihlášení přes Google na webu mělo fungovat.
