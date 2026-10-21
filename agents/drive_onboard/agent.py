"""
Drive onboarding agent — definition as code (Phase 3.3).

Creates or updates the Managed Agents agent + environment idempotently and
stores their ids plus a `prompt_version` (a hash of the system prompt + tool
schemas) in drive-onboard/agent.json. Every proposal records the prompt_version
that produced it, so a change to the prompt or tools is traceable.

    python -m agents.drive_onboard.agent sync      # create/update, bump prompt_version
    python -m agents.drive_onboard.agent show      # print the stored config

Requires an anthropic SDK with Managed Agents (>= 1.15; the repo-wide 1.3 pin
does NOT have it — the onboard-job image pins it explicitly) and ANTHROPIC_API_KEY.
"""

import hashlib
import json
import pathlib
import sys

from agents.drive_onboard.tools import TOOLS
from src.modules import onboarding as ob

MODEL = 'claude-sonnet-4-6'
AGENT_NAME = 'Drive Onboarding'
ENV_NAME = 'drive-onboard-env'

_PROMPT_PATH = pathlib.Path(__file__).with_name('system_prompt.md')


def system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding='utf-8')


def prompt_version() -> str:
    """Hash over the system prompt + tool schemas — changes iff the agent's
    behaviour definition changes."""
    h = hashlib.sha256()
    h.update(system_prompt().encode('utf-8'))
    h.update(json.dumps(TOOLS, sort_keys=True, default=str).encode('utf-8'))
    return h.hexdigest()[:16]


def _gcs():
    from google.cloud import storage
    return storage.Client()


def _anthropic():
    import anthropic
    client = anthropic.Anthropic()
    if not hasattr(client.beta, 'agents'):
        raise SystemExit(
            f'anthropic {anthropic.__version__} has no Managed Agents surface — '
            'the onboard-job image must pin anthropic >= 1.15.')
    return client


def sync(gcs_client=None, anth=None) -> dict:
    gcs_client = gcs_client or _gcs()
    anth = anth or _anthropic()
    cfg = ob.load_agent_config(gcs_client)
    pv = prompt_version()

    # Environment — create once, reuse.
    if not cfg.get('env_id'):
        env = anth.beta.environments.create(
            name=ENV_NAME,
            config={'type': 'cloud', 'networking': {'type': 'unrestricted'}})
        cfg['env_id'] = env.id
        print(f'created environment {env.id}', flush=True)

    # Agent — create once, then update in place when the definition changes.
    if not cfg.get('agent_id'):
        agent = anth.beta.agents.create(
            name=AGENT_NAME, model=MODEL, system=system_prompt(), tools=TOOLS)
        cfg.update(agent_id=agent.id, agent_version=agent.version)
        print(f'created agent {agent.id} v{agent.version}', flush=True)
    elif cfg.get('prompt_version') != pv:
        agent = anth.beta.agents.update(
            cfg['agent_id'], system=system_prompt(), tools=TOOLS)
        cfg['agent_version'] = agent.version
        print(f'updated agent {cfg["agent_id"]} -> v{agent.version}', flush=True)
    else:
        print(f'agent {cfg["agent_id"]} already at prompt_version {pv}', flush=True)

    cfg.update(prompt_version=pv, model=MODEL, name=AGENT_NAME)
    ob.save_agent_config(gcs_client, cfg)
    return cfg


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'show'
    if cmd == 'sync':
        print(json.dumps(sync(), indent=2, default=str))
    elif cmd == 'show':
        print(json.dumps(ob.load_agent_config(_gcs()), indent=2, default=str))
        print(f'current prompt_version would be: {prompt_version()}')
    else:
        print('usage: python -m agents.drive_onboard.agent [sync|show]', file=sys.stderr)
        sys.exit(2)
