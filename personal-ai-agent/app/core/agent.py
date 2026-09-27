"""Agent orchestrator: decides when to call LLM vs tools, maintains conversation history."""

from __future__ import annotations

from typing import Any

from app.core.llm_client import LLMClient
from app.tools.registry import execute_tool, tool_schemas
from app.memory.vector_store import VectorMemory  # will be created next


class Agent:
    """Core agent that loops: LLM → (tool use) → LLM → ... until final text."""

    def __init__(self) -> None:
        self.llm = LLMClient()
        self.history: list[dict] = []  # short-term memory (chat turns)
        self.vector_memory = VectorMemory()  # long-term semantic memory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, user_input: str) -> str:
        """Entry point: receive user text, return agent reply."""
        # 1. Append user message to short-term history
        self.history.append({"role": "user", "content": user_input})

        # 2. Also store in long-term memory (optional, but good for RAG)
        self.vector_memory.add(
            text=user_input,
            metadata={"role": "user", "turn": len(self.history)},
            doc_id=f"user_{len(self.history)}",
        )

        # 3. Tool-use loop (max iterations to avoid infinite loops)
        for _ in range(settings.max_iterations):
            # Get LLM response (may include tool_use blocks)
            response = self.llm.generate(self.history, tools=tool_schemas())

            # If LLM wants to call tools, execute them and continue loop
            tool_calls = self.llm.extract_tool_calls(response)
            if tool_calls:
                # Append assistant message with tool_use blocks
                self.history.append(self.llm.build_assistant_message(response))
                # For each tool call, execute and append tool_result
                for tc in tool_calls:
                    try:
                        result = execute_tool(tc["name"], tc["input"])
                        result_str = str(result)
                    except Exception as exc:  # pragma: no cover - defensive
                        result_str = f"Error executing tool {tc['name']}: {exc}"
                    self.history.append(
                        self.llm.build_tool_result_message(tc["id"], result_str)
                    )
                    # Also store tool result in long-term memory (optional)
                    self.vector_memory.add(
                        text=f"Tool {tc['name']} result: {result_str}",
                        metadata={"role": "tool_result", "tool": tc["name"]},
                        doc_id=f"tool_{len(self.history)}",
                    )
                # Continue loop to let LLM ingest tool results
                continue

            # No tool calls → we have final text
            final_text = self.llm.extract_text(response)
            # Append final assistant message to history
            self.history.append({"role": "assistant", "content": final_text})
            # Store in long-term memory
            self.vector_memory.add(
                text=final_text,
                metadata={"role": "assistant", "turn": len(self.history)},
                doc_id=f"assistant_{len(self.history)}",
            )
            return final_text

        # If we exit the loop, something went wrong (max iterations)
        return "I'm sorry, but I'm having trouble processing that request right now."

    # ------------------------------------------------------------------
    # Convenience for external callers (e.g., testing)
    # ------------------------------------------------------------------
    def get_history(self) -> list[dict]:
        return self.history.copy()

    # ------------------------------------------------------------------
    # Expose registry helpers for the streaming endpoint
    # ------------------------------------------------------------------
    def tools_schemas(self) -> list[dict]:
        return tool_schemas()

    def tools_execute(self, name: str, input_data: dict) -> Any:
        return execute_tool(name, input_data)


# lazy import to avoid circular dependency at module level
from config.settings import settings