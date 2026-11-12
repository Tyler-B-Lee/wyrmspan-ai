import json
from agent_tools import CardLookupTool


class WyrmspanReActAgent:
    def __init__(
        self,
        env,
        rag_tool,
        llm_client,
        model_name="gpt-4o",
        card_lookup_tool=None,
        max_react_steps=6,
    ):
        self.env = env
        self.rag_tool = rag_tool
        self.llm_client = llm_client
        self.model_name = model_name
        self.card_lookup_tool = card_lookup_tool if card_lookup_tool is not None else CardLookupTool()
        if max_react_steps < 1:
            raise ValueError("max_react_steps must be at least 1.")
        self.max_react_steps = max_react_steps

    def run_turn(self, raw_obs: dict) -> tuple[dict, float, bool, bool, dict]:
        """
        Runs a single turn loop for the agent. The agent can query tools
        multiple times if needed before finally submitting a game action.
        """
        # 1. Serialize environment observation into a readable briefing
        state_summary = self._serialize_observation(raw_obs)

        # Maintain a conversation history for the current turn's ReAct steps
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert strategist playing the board game Wyrmspan. "
                    "Your goal is to defeat the solo mode 'automa' opponent by outscoring it. In general, you want to maximize victory points (VP). "
                    "You have access to a rulebook search tool ('search_rulebook'). If you are unsure about "
                    "any game mechanics, costs, or interaction rules, you MUST query the tool before acting. "
                    "You can look up card details with the 'lookup_card' tool by providing "
                    "card_type ('dragon' or 'cave') and card_id (the card's number). "
                    "Request a tool with JSON shaped like "
                    '{"type":"tool_call","tool":"search_rulebook","query":"..."} or '
                    '{"type":"tool_call","tool":"lookup_card","card_type":"dragon|cave","card_id":1}. '
                    "Choose only from the legal actions listed in the state summary. "
                    "When ready, respond with JSON in this format: "
                    '{"type":"action","action":<legal action index>}. '
                    "The action value must be an integer index, not an action description."
                )
            },
            {"role": "user", "content": f"Current Game State:\n{state_summary}"}
        ]

        for step in range(self.max_react_steps):
            response = self._call_llm(messages=messages)

            # Case A: The LLM wants to call a tool
            if response.get("type") == "tool_call":
                tool_name = response.get("tool")
                query = response.get("query")

                if tool_name == "search_rulebook":
                    if not isinstance(query, str) or not query.strip():
                        tool_result = "Error: search_rulebook requires a non-empty query."
                    elif self.rag_tool is None:
                        tool_result = "Error: Rulebook search is not configured."
                    else:
                        tool_result = self.rag_tool.search_rules(query, top_k=2)
                elif tool_name == "lookup_card":
                    card_type = response.get("card_type")
                    card_id = response.get("card_id")
                    tool_result = self.card_lookup_tool.lookup_card(card_type, card_id)
                    query = f"{card_type} card #{card_id}"
                else:
                    tool_result = f"Error: Unknown tool '{tool_name}'."

                print(f"[Agent Tool Call]: {tool_name} ({query})")
                # Feed tool output back into the agent's thought process
                messages.append({"role": "assistant", "content": json.dumps(response)})
                messages.append({
                    "role": "user",
                    "content": f"Tool Result for '{query}':\n{tool_result}\n\nNow, continue your reasoning or make your final action."
                })

            # Case B: The LLM decides to take a game action (End of turn loop)
            elif response.get("type") == "action":
                action_idx = response.get("action")
                if isinstance(action_idx, bool) or not isinstance(action_idx, int):
                    raise ValueError(f"Agent action must be an integer index, got {action_idx!r}.")

                action_mask = raw_obs.get("action_mask", [])
                if hasattr(action_mask, "tolist"):
                    action_mask = action_mask.tolist()
                if action_idx < 0 or action_idx >= len(action_mask) or not action_mask[action_idx]:
                    raise ValueError(f"Agent selected illegal action index {action_idx}.")

                print(f"[Agent Action Chosen]: {action_idx}")

                # Step the Gymnasium environment using the step function
                next_obs, reward, terminated, truncated, info = self.env.step(action_idx)
                return next_obs, reward, terminated, truncated, info

        raise RuntimeError("Agent exceeded maximum ReAct reasoning steps without choosing an action.")

    def _serialize_observation(self, obs: dict) -> str:
        """
        Creates a compact default summary of observation fields and legal actions.
        """
        def flatten(value):
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                for item in value:
                    yield from flatten(item)
            else:
                yield value

        def readable_tokens(token_ids, token_mask=None):
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if hasattr(token_mask, "tolist"):
                token_mask = token_mask.tolist()

            vocabulary = getattr(self.env, "token_strings", [])
            tokens = []
            for position, token_id in enumerate(token_ids):
                if token_mask is not None and not token_mask[position]:
                    continue
                if not isinstance(token_id, int):
                    continue
                tokens.append(vocabulary[token_id] if 0 <= token_id < len(vocabulary) else str(token_id))
            return " ".join(tokens) if tokens else "(empty action)"

        lines = ["Observation fields:"]
        for key, value in obs.items():
            if key in {"action_token_ids", "action_token_mask", "action_mask"}:
                continue
            shape = getattr(value, "shape", None)
            values = list(flatten(value))
            preview = ", ".join(str(item) for item in values[:12])
            if len(values) > 12:
                preview += f", ... ({len(values)} values total)"
            shape_text = f" shape={tuple(shape)}" if shape is not None else ""
            lines.append(f"- {key}{shape_text}: [{preview}]")

        action_mask = obs.get("action_mask", [])
        if hasattr(action_mask, "tolist"):
            action_mask = action_mask.tolist()
        action_ids = obs.get("action_token_ids", [])
        action_masks = obs.get("action_token_mask", [])
        if hasattr(action_ids, "tolist"):
            action_ids = action_ids.tolist()
        if hasattr(action_masks, "tolist"):
            action_masks = action_masks.tolist()

        lines.append("Legal actions:")
        legal_indices = [index for index, valid in enumerate(action_mask) if valid]
        if not legal_indices:
            lines.append("- None")
        else:
            for index in legal_indices:
                token_mask = action_masks[index] if index < len(action_masks) else None
                action_tokens = action_ids[index] if index < len(action_ids) else []
                lines.append(f"- {index}: {readable_tokens(action_tokens, token_mask)}")

        return "\n".join(lines)

    def _call_llm(self, messages: list) -> dict:
        """
        Call an injected provider and parse its JSON decision.

        Supports an OpenAI-compatible Chat Completions client or a callable
        adapter accepting ``messages`` and ``model`` keyword arguments. Callable
        adapters may return a dict directly or a JSON string.
        """
        if self.llm_client is None:
            raise RuntimeError("An LLM client must be provided to choose an action.")

        if callable(self.llm_client):
            result = self.llm_client(messages=messages, model=self.model_name)
        else:
            completions = getattr(getattr(self.llm_client, "chat", None), "completions", None)
            create = getattr(completions, "create", None)
            if not callable(create):
                raise TypeError(
                    "llm_client must be callable or expose chat.completions.create()."
                )
            result = create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )

        if isinstance(result, dict):
            response_data = result
        else:
            content = result if isinstance(result, str) else None
            if content is None:
                try:
                    content = result.choices[0].message.content
                except (AttributeError, IndexError, TypeError) as error:
                    raise ValueError("LLM response did not contain JSON decision content.") from error
            try:
                response_data = json.loads(content)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"LLM response must be a JSON object; received {content!r}.") from error

        if not isinstance(response_data, dict):
            raise ValueError("LLM response JSON must be an object.")
        if response_data.get("type") not in {"tool_call", "action"}:
            raise ValueError("LLM response type must be 'tool_call' or 'action'.")
        return response_data