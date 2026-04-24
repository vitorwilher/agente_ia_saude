import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shiny.express import ui
from shiny.session import get_current_session

from agent import invoke_agent
from globals import configs

ui.page_opts(
    title=ui.div(
        ui.h3(configs["ui"]["app_title"]),
        ui.input_dark_mode(mode="dark"),
        class_="d-flex justify-content-between w-100",
    ),
    window_title=configs["ui"]["app_title"],
    fillable=True,
    fillable_mobile=True,
)

chat = ui.Chat(id="chat")
chat.ui(messages=configs["chat"]["welcome_messages"])


@chat.on_user_submit
async def handle_user_input(user_input: str):
    session = get_current_session()
    thread_id = session.id if session else "default"
    response = invoke_agent(user_input, thread_id=thread_id)
    await chat.append_message(response)
