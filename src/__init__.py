# src/__init__.py

# Import the main agent instance from the coding_agent module
# and assign it to the variable 'agent' so the ADK runner can find it.
from .coding_agent import main_agent as agent

print(f"Loaded agent '{agent.name}' from src/__init__.py") 