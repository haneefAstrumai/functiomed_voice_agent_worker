import inspect
from livekit.agents import AgentSession
print("AgentSession annotations:", getattr(AgentSession, '__annotations__', {}))
import livekit.agents
print("dir(livekit.agents):", dir(livekit.agents))
