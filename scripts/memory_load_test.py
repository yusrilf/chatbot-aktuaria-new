#!/usr/bin/env python3
"""Memory load test: push 150 Q/A messages without calling OpenAI.

This script exercises EnhancedActuarialChatService's session memory by directly
saving random question/answer pairs, verifying that pruning keeps memory bounded.
"""
import os
import sys
import random
import string
import time

# Ensure project root is on sys.path for 'app' imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Ensure an OPENAI_API_KEY is present to avoid import-time errors
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-memory-test")

from app.services.enhanced_chat_service import EnhancedActuarialChatService
from app.config import config


def random_text(n: int = 64) -> str:
    alphabet = string.ascii_letters + string.digits + " "
    return "".join(random.choice(alphabet) for _ in range(n))


def main():
    print("=== Memory Load Test (150 messages, no OpenAI calls) ===")
    service = EnhancedActuarialChatService()
    session_id = "memory-test"

    # Ensure session memory exists
    service._ensure_session_memory(session_id)
    memory = service.session_memories[session_id]

    # Report initial state
    k = getattr(config, "MEMORY_WINDOW_SIZE", 6)
    max_messages = k * 2
    print(f"Configured MEMORY_WINDOW_SIZE (k): {k} -> max_messages kept: {max_messages}")
    print(f"Initial messages: {len(memory.chat_memory.messages)}")

    # Push 150 Q/A interactions
    start = time.time()
    for i in range(150):
        q = f"Q{i}: " + random_text(50)
        a = f"A{i}: " + random_text(60)
        # Directly save to memory (no LLM invocation)
        service._save_to_memory(session_id, q, a)
        if (i + 1) % 25 == 0:
            print(f"After {i+1} interactions -> messages: {len(memory.chat_memory.messages)}")

    duration = time.time() - start

    # Final checks
    final_len = len(memory.chat_memory.messages)
    print(f"Final messages count: {final_len} (expected <= {max_messages})")
    print(f"Elapsed time: {duration:.3f}s")

    # Show formatted chat history string (should reflect last k turns)
    history_str = service._get_chat_history_string(session_id)
    print("\n=== Chat History String (bounded) ===")
    print(history_str)

    # Verify conversation history API returns recent subset only
    conversations = service.get_conversation_history(session_id=session_id, limit=10)
    print(f"\nConversations returned: {len(conversations)} (bounded by available pairs)")
    if conversations:
        print("Last conversation sample:")
        last = conversations[-1]
        print({k: (v[:80] + "...") if isinstance(v, str) and len(v) > 80 else v for k, v in last.items()})


if __name__ == "__main__":
    main()