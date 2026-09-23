"""
Reading a Claude response safely
--------------------------------
`resp.content[0].text` is the obvious way to get the text out of an Anthropic
response, and it is wrong in two ways that both fail *confusingly*:

  * **An empty content list raises IndexError.** Claude returns zero content
    blocks when it declines to answer (`stop_reason='refusal'`), and can in
    principle return none for other terminal stop reasons. IndexError is not
    the ValueError that retry/parse loops in this codebase catch, so it escapes
    the loop that was supposed to handle a bad response and surfaces as
    `list index out of range` — a message that names neither the model, the
    company, nor the refusal. That shipped: a real client (aerosolised
    thermostable vaccine powders, DoD CBD SBIR on an MVA smallpox/mpox vaccine)
    was unbuildable for two runs with no usable diagnosis.
  * **Block 0 is not guaranteed to be text.** Any response that leads with a
    non-text block breaks the assumption, silently on some SDK versions.

So: join every text block, and turn "no text at all" into an error that carries
`stop_reason`, which is the field that actually explains it.

Deliberately stdlib-only and Streamlit-free — imported by views, by the jobs,
and by src/modules, so it must stay importable from every image. Each job image
COPYs it explicitly (only jobs/Dockerfile takes all of src/).
"""


class EmptyResponseError(ValueError):
    """Claude returned no text. Subclasses ValueError so the existing
    `except ValueError` retry/parse handlers in this codebase keep working, and
    is named so `type(e).__name__` is self-explaining where a failure reason is
    reported back to the user (see aspect_matching.rerank_async)."""


def response_text(resp, *, default: str | None = None) -> str:
    """The text of a Claude response.

    Raises EmptyResponseError when there is none — pass `default` only where an
    empty answer is genuinely survivable and the caller has somewhere sane to
    put it, never to make a failure quiet.
    """
    blocks = getattr(resp, 'content', None) or []
    text = ''.join(
        getattr(block, 'text', '') or ''
        for block in blocks
        if getattr(block, 'type', '') == 'text'
    ).strip()
    if text:
        return text
    if default is not None:
        return default
    raise EmptyResponseError(
        f'Claude returned no text (stop_reason='
        f'{getattr(resp, "stop_reason", None)}, {len(blocks)} content block(s))'
    )
