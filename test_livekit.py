import inspect
from livekit.agents import AgentSession, Agent
print("Agent methods:", [m for m in dir(Agent) if not m.startswith('_')])
print("AgentSession methods:", [m for m in dir(AgentSession) if not m.startswith('_')])
