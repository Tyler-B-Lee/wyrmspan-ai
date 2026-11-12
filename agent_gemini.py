import json
from agent_tools import CardLookupTool, RulebookSearchTool
from google import genai
from pydantic import BaseModel, Field

# Answer Structure
class ActionResponse(BaseModel):
    action: int = Field(description="The index of the legal action to take, from the actions listed in the current turn's description.")
    reasoning: str = Field(description="The reasoning behind the action chosen.")

# Function Declarations
search_rulebook_declaration = {
    "type": "function",
    "name": "search_rulebook",
    "description": "Search the Wyrmspan rulebook for relevant rules.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to search the rulebook for.",
            }
        },
        "required": ["query"],
    },
}

card_lookup_declaration = {
    "type": "function",
    "name": "lookup_card",
    "description": "Look up a Wyrmspan dragon or cave card by its type and number.",
    "parameters": {
        "type": "object",
        "properties": {
            "card_type": {
                "type": "string",
                "enum": ["dragon", "cave"],
                "description": "The type of card to look up.",
            },
            "card_id": {
                "type": "integer",
                "description": "The number of the card to look up. Dragons are 1-183, while caves are 1-75.",
            }
        },
        "required": ["card_type", "card_id"],
    },
}


if __name__ == "__main__":
    client = genai.Client()

    function_tools = {
        "search_rulebook": RulebookSearchTool().search_rules,
        "lookup_card": CardLookupTool().lookup_card,
    }

    prompt = """
        You are an expert strategist playing the board game Wyrmspan.
        Your goal is to defeat the solo mode 'automa' opponent by outscoring it. In general, you want to maximize victory points (VP).
        You have access to a rulebook search tool ('search_rulebook'). If you are unsure about
        any game mechanics, costs, or interaction rules, you MUST query the appropriate tool before acting.
        Choose only from the legal actions listed in the state summary.
        When ready, respond with JSON in this format:
        {"type":"action","action":<legal action index>}.
        The action value must be an integer index, not an action description.
    """

    previous_id = None
    while True:
        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt,
            tools=[
                search_rulebook_declaration,
                card_lookup_declaration,
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": ActionResponse.model_json_schema(),
            },
            previous_interaction_id=previous_id,
        )

        chosen_action = None
        for step in interaction.steps:
            if step.type == "function_call":
                result = function_tools[step.name](**step.arguments)
                print(f"Called {step.name} with arguments {step.arguments}")
                print(f"> Result: {result}")
                