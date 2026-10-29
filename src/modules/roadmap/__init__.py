"""
roadmap — the event-kiosk funding-roadmap feature (Stage 17).

A founder fills a short HubSpot form (name, email, phone, company URL, a 10-50
word "what does your company do", and an optional pitch-deck PDF). A workflow
webhook hits intake-web, which builds a capability profile from the description
(and deck), matches grant topics, renders a branded roadmap PDF, hosts it, and
writes its URL back to the HubSpot contact so HubSpot's own email/text sequence
can send it.

Streamlit-free. Shared, trusted cores are reused rather than reimplemented:
profile_build (profile from text), aspect_matching (score + re-rank), the
intake service's HubSpot client, doc_extract (deck text), and GoogleBucketManager.
"""
