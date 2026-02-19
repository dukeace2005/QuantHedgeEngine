import streamlit as st
import os
import json
import inspect
import pickle
from datetime import datetime
from pathlib import Path
from schwab.auth import (
    client_from_token_file,
    client_from_received_url,
    get_auth_context,
)

class SchwabInterface:
    def __init__(self):
        # Load secrets from streamlit secrets
        if "schwab" not in st.secrets:
            st.error("Missing [schwab] section in secrets.toml.")
            self.api_key = None
            self.app_secret = None
            self.redirect_url = None
        else:
            self.api_key = st.secrets["schwab"]["api_key"]
            self.app_secret = st.secrets["schwab"]["app_secret"]
            self.redirect_url = st.secrets["schwab"].get("redirect_url")

            if not self.redirect_url:
                st.error("Missing schwab.redirect_url in secrets.toml.")

        self.token_path = str(Path(__file__).resolve().parent / ".streamlit" / "schwab_token.json")
        self.auth_context_path = str(Path(__file__).resolve().parent / ".streamlit" / "schwab_auth_context.pkl")
        self.debug_log_path = str(Path(__file__).resolve().parent / ".streamlit" / "auth_debug.log")
        self.client = None
        self.auth_context = st.session_state.get("schwab_auth_context")
        if self.auth_context is None:
            self.auth_context = self._load_auth_context()
            if self.auth_context is not None:
                st.session_state["schwab_auth_context"] = self.auth_context

    def _log_debug(self, message):
        try:
            log_path = Path(self.debug_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} | {message}\n")
        except Exception:
            pass

    def _save_auth_context(self):
        try:
            if self.auth_context is None:
                return
            path = Path(self.auth_context_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self.auth_context, f)
            self._log_debug(f"auth_context_saved path={self.auth_context_path}")
        except Exception as e:
            self._log_debug(f"auth_context_save_failed={str(e)}")

    def _load_auth_context(self):
        try:
            path = Path(self.auth_context_path)
            if not path.exists():
                return None
            with open(path, "rb") as f:
                ctx = pickle.load(f)
            self._log_debug(f"auth_context_loaded path={self.auth_context_path}")
            return ctx
        except Exception as e:
            self._log_debug(f"auth_context_load_failed={str(e)}")
            return None

    def _clear_auth_context(self):
        try:
            Path(self.auth_context_path).unlink(missing_ok=True)
            self._log_debug("auth_context_cleared")
        except Exception as e:
            self._log_debug(f"auth_context_clear_failed={str(e)}")

    def get_auth_url(self):
        """Generate the URL for the user to authorize the app."""
        if not self.api_key or not self.app_secret or not self.redirect_url:
            return "#"
        if self.auth_context is None:
            self.auth_context = get_auth_context(self.api_key, self.redirect_url)
            st.session_state["schwab_auth_context"] = self.auth_context
            self._save_auth_context()
        return self.auth_context.authorization_url

    def authenticate_from_token(self):
        """Attempt to authenticate using a stored token file."""
        if not self.api_key or not self.app_secret or not self.redirect_url:
            return False, "Missing Schwab config in secrets.toml."

        if not os.path.exists(self.token_path):
            return False, "No token found. Please authenticate."
        
        try:
            # client_from_token_file handles token refresh automatically
            self.client = client_from_token_file(self.token_path, self.api_key, self.app_secret)
            return True, "Connected via stored token"
        except Exception as e:
            return False, f"Token expired or invalid: {str(e)}"

    def authenticate_from_url(self, response_url):
        """Complete the auth flow using the redirect URL pasted by the user."""
        try:
            if not self.api_key or not self.app_secret or not self.redirect_url:
                return False, "Missing Schwab config in secrets.toml."

            if self.auth_context is None:
                self.auth_context = self._load_auth_context()
                if self.auth_context is not None:
                    st.session_state["schwab_auth_context"] = self.auth_context
                else:
                    self._log_debug("missing_auth_context_for_callback")
                    return False, "Missing auth context for callback. Start authorization again."

            def _write_token(token, *args, **kwargs):
                Path(self.token_path).parent.mkdir(parents=True, exist_ok=True)
                payload = token
                # Handle token objects that are not directly JSON serializable.
                if hasattr(payload, "to_json"):
                    payload = payload.to_json()
                elif not isinstance(payload, (dict, list, str, int, float, bool, type(None))):
                    payload = getattr(payload, "__dict__", str(payload))

                with open(self.token_path, "w", encoding="utf-8") as f:
                    if isinstance(payload, str):
                        f.write(payload)
                    else:
                        json.dump(payload, f)
                self._log_debug(f"token_write_callback_called path={self.token_path}")

            # Support both schwab-py variants:
            # - token write callback argument
            # - token path argument
            sig = inspect.signature(client_from_received_url)
            param_names = list(sig.parameters.keys())

            if "token_write_func" in param_names:
                self._log_debug("client_from_received_url using token_write_func")
                self.client = client_from_received_url(
                    self.api_key,
                    self.app_secret,
                    self.auth_context,
                    response_url,
                    token_write_func=_write_token
                )
            elif "token_path" in param_names:
                Path(self.token_path).parent.mkdir(parents=True, exist_ok=True)
                self._log_debug("client_from_received_url using token_path kwarg")
                self.client = client_from_received_url(
                    self.api_key,
                    self.app_secret,
                    self.auth_context,
                    response_url,
                    token_path=self.token_path
                )
            else:
                # Fallback for older positional signatures
                Path(self.token_path).parent.mkdir(parents=True, exist_ok=True)
                self._log_debug("client_from_received_url using positional token_path")
                self.client = client_from_received_url(
                    self.api_key,
                    self.app_secret,
                    self.auth_context,
                    response_url,
                    self.token_path
                )

            if not os.path.exists(self.token_path):
                self._log_debug("auth_completed_but_no_token_file")
                return False, "Authentication succeeded but token was not persisted."

            self._log_debug("token_file_present_after_auth")
            st.session_state.pop("schwab_auth_context", None)
            self._clear_auth_context()
            return True, "Authentication successful! Token saved."
        except Exception as e:
            self._log_debug(f"authenticate_from_url_exception={str(e)}")
            return False, f"Authentication failed: {str(e)}"
