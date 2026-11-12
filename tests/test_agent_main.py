from types import SimpleNamespace

from agent_main import WyrmspanReActAgent


class FakeEnv:
    token_strings = ["<pad>", "pass"]

    def __init__(self):
        self.action_index = None

    def step(self, action_index):
        self.action_index = action_index
        return {"next": True}, 0.5, False, False, {"chosen_action": action_index}


class FakeRulebook:
    def __init__(self):
        self.query = None

    def search_rules(self, query, top_k):
        self.query = (query, top_k)
        return "The rulebook result."


def test_run_turn_uses_tool_result_then_submits_legal_action():
    env = FakeEnv()
    rulebook = FakeRulebook()
    responses = [
        {"type": "tool_call", "tool": "search_rulebook", "query": "guild scoring"},
        {"type": "action", "action": 0},
    ]
    calls = []

    def fake_client(messages, model):
        calls.append((messages, model))
        return responses.pop(0)

    agent = WyrmspanReActAgent(env, rulebook, fake_client)
    observation = {
        "player_resources": [2, 1],
        "action_mask": [1, 0],
        "action_token_ids": [[1], [0]],
        "action_token_mask": [[1], [0]],
    }

    result = agent.run_turn(observation)

    assert result == ({"next": True}, 0.5, False, False, {"chosen_action": 0})
    assert env.action_index == 0
    assert rulebook.query == ("guild scoring", 2)
    assert len(calls) == 2
    assert "The rulebook result." in calls[1][0][-1]["content"]


def test_call_llm_parses_openai_compatible_chat_response():
    request = {}

    def create(**kwargs):
        request.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"type":"action","action":0}')
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    agent = WyrmspanReActAgent(FakeEnv(), None, client)

    response = agent._call_llm([{"role": "user", "content": "Choose."}])

    assert response == {"type": "action", "action": 0}
    assert request["model"] == "gpt-4o"
    assert request["response_format"] == {"type": "json_object"}