import asyncio
import textwrap

from langchain_core.messages import SystemMessage, AnyMessage


def get_leading_prompts() -> list[AnyMessage]:
    # chat_prompt_template = await asyncio.to_thread(client.pull_prompt, f'blahblah')
    # template_args = {
    # }
    # messages = await chat_prompt_template.aformat_messages(**template_args)
    messages = [
        SystemMessage(content=textwrap.dedent(f"""
            You are a helpful assistant who wants to make sure the user is comfortable
            with the cloths they choose each morning to get dressed.
            
            Assume the user is living in Columbia MO in the United States. For the national weather service,
            that location, gridId is "LSX", x is "20", and y is "87".
        """))
    ]
    return messages