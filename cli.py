"""Run TasteProof from the command line: python cli.py "Plan a Friday night in Austin for fans of Phoebe Bridgers" [--fault]"""
import json
import sys

from tasteproof.agent import Agent
from tasteproof.llm import GrokLLM

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--fault"]
    q = " ".join(args) or "Plan a Friday night in Austin for fans of Phoebe Bridgers"
    r = Agent(GrokLLM(), inject_fault="--fault" in sys.argv).run(q)
    print(json.dumps({k: r[k] for k in ("qloo_mode", "steps", "itinerary", "attempts", "dropped", "tool_counts", "elapsed_s", "qloo_requests")}, indent=2))
