---
name: OAuth Authorization Code flow gap
description: URC engine lacks OAuth Authorization Code flow — needed for ThousandEyes, Microsoft Graph, Salesforce, etc.
type: project
---

URC currently supports OAuth2 Client Credentials flow only. Authorization Code flow (interactive login + MFA) is missing.

**Why:** Many modern APIs (Cisco ThousandEyes, Microsoft Graph, Salesforce) require Authorization Code flow. The Splunk TA for ThousandEyes already supports this — UCC handles it natively at the platform level (popup login, redirect callback via Splunk Web, token storage in passwords.conf). Users expect parity.

**How to apply:** Plan this into a future iteration. The work involves:
1. Add an `oauth` entity type option in globalConfig account tab (UCC renders the login popup natively)
2. Ensure the engine can read tokens stored by UCC's OAuth flow from Splunk's credential store
3. Handle token refresh using the stored refresh_token (OAuthAuthenticator already has refresh logic — extend it)
4. UCC does the heavy lifting (popup, redirect, code-for-token exchange) — our work is mostly wiring config and token retrieval
