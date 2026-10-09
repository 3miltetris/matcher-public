"""
mint_gmail_token.py — one-time helper to mint the Gmail OAuth refresh token for
the shared 2FA mailbox (Stage 10 email-OTP).

Run this ONCE, locally, signed in as the shared mailbox account. It prints the
JSON blob to store in Secret Manager as `funding-2fa-gmail-oauth`, which the
deep-research-job reads to pull one-time 2FA codes out of that mailbox.

This is NOT part of any container image — it needs a desktop browser for the
OAuth consent, and `google-auth-oauthlib` which the images do not carry.

Prerequisites
-------------
1. OAuth consent screen for project cc-matcher-v1 set to **Internal** (so the
   restricted gmail.readonly scope needs no Google verification and the refresh
   token does not expire).
2. An OAuth **Desktop** client created under that project; download its JSON as
   `client_secret.json`.
3. A virtualenv with: pip install google-auth-oauthlib

Usage
-----
    python scripts/mint_gmail_token.py --client-secret client_secret.json \
        --mailbox funding-2fa@bwcoconsulting.com > token.json

    gcloud secrets create funding-2fa-gmail-oauth \
        --data-file=token.json --project cc-matcher-v1
    # (or: gcloud secrets versions add funding-2fa-gmail-oauth --data-file=token.json ...)

A browser window opens for you to sign in as the shared mailbox and grant
read-only Gmail access. Delete token.json and client_secret.json afterwards.
"""

import argparse
import json
import sys

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def main() -> int:
    ap = argparse.ArgumentParser(description='Mint the 2FA-mailbox Gmail refresh token.')
    ap.add_argument('--client-secret', required=True,
                    help='Path to the Desktop OAuth client JSON (client_secret.json).')
    ap.add_argument('--mailbox', required=True,
                    help='The shared mailbox address (stored for display only).')
    ap.add_argument('--port', type=int, default=0,
                    help='Local redirect port (0 = pick a free one).')
    args = ap.parse_args()

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print('google-auth-oauthlib is required: pip install google-auth-oauthlib',
              file=sys.stderr)
        return 1

    flow = InstalledAppFlow.from_client_secrets_file(args.client_secret, SCOPES)
    # access_type=offline + consent prompt guarantees a refresh token is returned.
    creds = flow.run_local_server(port=args.port, access_type='offline',
                                  prompt='consent')

    if not creds.refresh_token:
        print('No refresh token returned — re-run and ensure you approve consent '
              '(revoke the prior grant at myaccount.google.com if needed).',
              file=sys.stderr)
        return 1

    blob = {
        'client_id':     creds.client_id,
        'client_secret': creds.client_secret,
        'refresh_token': creds.refresh_token,
        'mailbox':       args.mailbox,
    }
    print(json.dumps(blob, indent=2))
    print('\nStore the JSON above as the secret funding-2fa-gmail-oauth.',
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
