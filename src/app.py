
from chatlas import ChatGoogle
from shiny.express import ui
from rag import invoke_qa
from dotenv import load_dotenv
from globals import configs
import os

# Define the Google Gemini chat client
_ = load_dotenv(configs["server"]["env_file_path"])
chat_client = ChatGoogle(
    api_key = os.environ.get(configs["chat"]["llm_model"]["api_env_var_name"]),
    system_prompt = configs["chat"]["system_prompt"],
    model = configs["chat"]["llm_model"]["name"]
)

# Set some Shiny page options
ui.page_opts(
    title = ui.div(
        ui.h3(configs["ui"]["app_title"]),
        ui.input_dark_mode(mode = "dark"),
        class_ = "d-flex justify-content-between w-100",
    ),
    window_title = configs["ui"]["app_title"],
    fillable = True,
    fillable_mobile = True
)

# Create and display empty chat
chat = ui.Chat(
    id = "chat",
    messages = configs["chat"]["welcome_messages"],
)
chat.ui()

# Handle user input and generate response
@chat.on_user_submit
async def handle_user_input(user_input: str):
    response = invoke_qa(user_input)
    await chat.append_message_stream(response)
