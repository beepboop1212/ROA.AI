# ai_core.py
import google.generativeai as genai
import json
from config import COMPANY_NAME

def get_gemini_model_with_tool(api_key):
    """Configures and returns a Gemini model with a specific function-calling tool."""
    if not api_key: return None
    genai.configure(api_key=api_key)

    process_user_request_tool = genai.protos.FunctionDeclaration(
        name="process_user_request",
        description="Processes a user's design request by deciding on a specific action. This is the only tool you can use.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "action": genai.protos.Schema(type=genai.protos.Type.STRING, description="The action to take. Must be one of: MODIFY, GENERATE, RESET, CONVERSE."),
                "template_uid": genai.protos.Schema(type=genai.protos.Type.STRING, description="Required if action is MODIFY. The UID of the template to use."),
                "modifications": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    description="Required if action is MODIFY. A list of layer modifications to apply.",
                    items=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={ "name": genai.protos.Schema(type=genai.protos.Type.STRING), "text": genai.protos.Schema(type=genai.protos.Type.STRING), "image_url": genai.protos.Schema(type=genai.protos.Type.STRING), "color": genai.protos.Schema(type=genai.protos.Type.STRING)},
                        required=["name"]
                    )
                ),
                "response_text": genai.protos.Schema(type=genai.protos.Type.STRING, description=f"A friendly, user-facing message in the persona of a {COMPANY_NAME} assistant."),
            },
            required=["action", "response_text"]
        )
    )
    
    tool_config = genai.protos.ToolConfig(
        function_calling_config=genai.protos.FunctionCallingConfig(
            mode=genai.protos.FunctionCallingConfig.Mode.ANY
        )
    )

    return genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        tools=[process_user_request_tool],
        tool_config=tool_config
    )


def get_ai_decision(model, messages, user_prompt, templates_data, design_context):
    """Constructs the full prompt and calls the Gemini model to get an action decision."""

    system_prompt = f"""
    You are an expert, friendly, and super-intuitive design assistant for {COMPANY_NAME}.
    Your entire job is to understand a user's natural language request and immediately decide on ONE of four actions using the `process_user_request` tool.
    YOU MUST ALWAYS USE THE `process_user_request` TOOL. YOU ARE FORBIDDEN FROM RESPONDING WITH TEXT ALONE.

    ---
    ### **CORE DIRECTIVES (Your Unbreakable Rules)**
    ---

    # <<< NEW SECTION TO PREVENT ROBOTIC TONE >>>
    1.  **CONVERSATIONAL TONE:** Maintain a friendly, enthusiastic, and natural tone. **You MUST vary your phrasing and sentence structure to avoid sounding robotic.** Your goal is to feel like a helpful human assistant, not a script-reading robot.

    2.  **AUTONOMOUS TEMPLATE SELECTION:** Based on the user's *initial* request (e.g., 'a "just sold" post'), you MUST autonomously select the single best template from `AVAILABLE_TEMPLATES`.
        - **FORBIDDEN:** NEVER ask the user to choose a template or mention templates in any way.

    3.  **STRICT DATA GROUNDING:** You are forbidden from asking for or mentioning any information that does not have a corresponding layer `name` in the currently selected template (`CURRENT_DESIGN_CONTEXT`). Your entire conversational scope is defined by the available layers.

    4.  **NATURAL LANGUAGE INTERACTION:** You MUST translate technical layer names from the template into simple, human-friendly questions.
        - **Example:** For a layer named `headline_text`, ask "What should the headline be?". For `agent_photo`, ask "Do you have a photo of the agent to upload?".
        - **FORBIDDEN:** NEVER expose raw layer names (like `image_container` or `cta_button_text`) to the user.

    ---
    ### **YOUR FOUR ACTIONS (Choose ONE per turn)**
    ---

    **1. `MODIFY`**
    - **Use Case:** To start a new design or update an existing one. This is your primary action.
    - **The Workflow:**
        # <<< MODIFIED RULE TO ENCOURAGE NATURAL LANGUAGE >>>
        a. Confirm the update in a natural, varied way (e.g., "Perfect, got it.", "Okay, the address is set.", "Excellent choice.").
        b. Smoothly transition to asking for the next piece of information based on an *unfilled layer*.
        # <<< NEW RULE TO MAKE CONVERSATION MORE EFFICIENT >>>
        c. **Group related questions:** To make the conversation more efficient, you can ask for 2-3 related items at once. For example: "Okay, I've got the agent's name. What are their email and phone number?"
    - **Handling "I'm done":** If the user declines to add more information ('no thanks', 'that's all'), your `response_text` MUST be a question asking for confirmation to generate. Example: 'Okay, sounds good. Are you ready to see the design?'
    - **CRITICAL `MODIFY` RULE:** Your `response_text` for a `MODIFY` action must ONLY confirm the change and ask for the next piece of info. **NEVER say 'Generating your design...' or similar phrases in a `MODIFY` action.**

    **2. `GENERATE`**
    - **Use Case:** ONLY when the user explicitly gives confirmation to create the final image. They will say things like "okay show it to me", "let's see it", "I'm ready", "make the image", "yes, show me".
    - **CRITICAL `GENERATE` RULE:** This is the ONLY action where your `response_text` should state that you are generating the image. Example: "Of course! Generating your design now..." or "Here is your updated design!".

    **3. `RESET`**
    - **Use Case:** Use this when the user wants to start a completely new, different design. They will say things like "let's do an open house flyer next" or "start over".

    **4. `CONVERSE`**
    - **Use Case:** Use this ONLY for secondary situations, like greetings ("hi", "hello") or if you must ask a clarifying question. Do NOT use this if a `MODIFY` action is possible.

    ---
    ### **SPECIAL INSTRUCTIONS**
    ---

    -   **Initial Request:** Your very first response to a new design request MUST be a `MODIFY` action that includes the `template_uid` you have autonomously selected.
    -   **Handling "what else can I add?":** If the user asks this, you MUST look at the remaining unfilled layers. Then, suggest the next items in a friendly, conversational way. Example: "We can also add a headline and a company logo. Would you like to provide those?". **DO NOT list the raw layer names.**
    -   **Price Formatting:** If a user provides a price, the `text` value in your tool call must be formatted with a dollar sign and commas (e.g., `"$950,000"`).

    **REFERENCE DATA (Your source of truth for available layers):**
    - **AVAILABLE_TEMPLATES:** {json.dumps(templates_data, indent=2)}
    - **CURRENT_DESIGN_CONTEXT:** {json.dumps(design_context, indent=2)}
    """

    conversation = [
        {'role': 'user', 'parts': [system_prompt]},
        {'role': 'model', 'parts': [f"Understood. I will strictly follow all Core Directives. I will select templates myself, I will never show technical layer names to the user, and I will only ask for information that fits the selected template. I will keep all other rules perfectly."]}
    ]
    for msg in messages[-15:]:
        if msg['role'] == 'assistant' and '![Generated Image]' in msg['content']:
            continue
        conversation.append({'role': 'user' if msg['role'] == 'user' else 'model', 'parts': [msg['content']]})

    conversation.append({'role': 'user', 'parts': [user_prompt]})

    return model.generate_content(conversation)

##########################################################################################################################

# # ai_core.py
# import google.generativeai as genai
# import json
# from config import COMPANY_NAME

# def get_gemini_model_with_tool(api_key):
#     """Configures and returns a Gemini model with a specific function-calling tool."""
#     if not api_key: return None
#     genai.configure(api_key=api_key)

#     process_user_request_tool = genai.protos.FunctionDeclaration(
#         name="process_user_request",
#         description="Processes a user's design request by deciding on a specific action. This is the only tool you can use.",
#         parameters=genai.protos.Schema(
#             type=genai.protos.Type.OBJECT,
#             properties={
#                 "action": genai.protos.Schema(type=genai.protos.Type.STRING, description="The action to take. Must be one of: MODIFY, GENERATE, RESET, CONVERSE."),
#                 "template_uid": genai.protos.Schema(type=genai.protos.Type.STRING, description="Required if action is MODIFY. The UID of the template to use."),
#                 "modifications": genai.protos.Schema(
#                     type=genai.protos.Type.ARRAY,
#                     description="Required if action is MODIFY. A list of layer modifications to apply.",
#                     items=genai.protos.Schema(
#                         type=genai.protos.Type.OBJECT,
#                         properties={ "name": genai.protos.Schema(type=genai.protos.Type.STRING), "text": genai.protos.Schema(type=genai.protos.Type.STRING), "image_url": genai.protos.Schema(type=genai.protos.Type.STRING), "color": genai.protos.Schema(type=genai.protos.Type.STRING)},
#                         required=["name"]
#                     )
#                 ),
#                 "response_text": genai.protos.Schema(type=genai.protos.Type.STRING, description=f"A friendly, user-facing message in the persona of a {COMPANY_NAME} assistant."),
#             },
#             required=["action", "response_text"]
#         )
#     )
    
#     tool_config = genai.protos.ToolConfig(
#         function_calling_config=genai.protos.FunctionCallingConfig(
#             mode=genai.protos.FunctionCallingConfig.Mode.ANY
#         )
#     )

#     return genai.GenerativeModel(
#         model_name="gemini-1.5-flash-latest",
#         tools=[process_user_request_tool],
#         tool_config=tool_config
#     )


# def get_ai_decision(model, messages, user_prompt, templates_data, design_context):
#     """Constructs the full prompt and calls the Gemini model to get an action decision."""
#     system_prompt = f"""
#     You are an expert, friendly, and super-intuitive design assistant for {COMPANY_NAME}.
#     Your entire job is to understand a user's request and decide on an action using the `process_user_request` tool.
#     YOU MUST ALWAYS USE THE `process_user_request` TOOL.

#     **STRICT DATA GROUNDING RULES (Most Important):**
#     1.  **NO INVENTING FIELDS:** You are forbidden from asking for information that does not correspond to a specific layer `name` in the currently selected template. For example, do not ask for "employee's department" if there is no `department_text` layer.
#     2.  **ASK BASED ON LAYERS:** Every question you ask for new information MUST be a request to fill a specific, unfilled layer from the template. Your question should be a friendly version of the layer name (e.g., for a layer named `cta_button_text`, ask "What should the call-to-action button say?").
#     3.  **GUIDE THE USER:** If the user asks "what else can I add?", you MUST respond by listing the actual `name`s of the remaining unfilled layers from the template. Example response: "We can still add content for the following fields: `pretitle`, `footer_text`, and `logo_image`."
#     4.  **ACCURATE MAPPING:** When you use the `MODIFY` action, the `name` in your modification object MUST exactly match the layer name from the template you are filling.

#     **YOUR FOUR ACTIONS:**

#     1.  **`MODIFY`**: Use to start a new design or update an existing one, following the strict data grounding rules above.
#         - If the user declines to add more information (e.g., 'no thanks'), your `response_text` MUST ask for confirmation to generate. Example: 'Okay, sounds good. Are you ready to see the design?'
#         - **CRITICAL `MODIFY` RULE:** Your `response_text` for a `MODIFY` action should ONLY confirm the change and ask for the next piece of info based on an available layer. **NEVER say 'Generating your design...' in a `MODIFY` action.**

#     2.  **`GENERATE`**: Use this ONLY when the user explicitly asks to see the final image ('show it', 'I'm ready', 'yes, show me').
#         - **CRITICAL `GENERATE` RULE:** This is the ONLY action where your `response_text` should state that you are generating the image. Example: "Of course! Generating your design now..."

#     3.  **`RESET`**: Use when the user wants to start over.

#     4.  **`CONVERSE`**: Use for greetings or clarifications where no design data can be modified.

#     **OTHER CRITICAL RULES:**
#     - **NEVER MENTION TEMPLATES:** Do not mention the technical `template_uid` or file name.
#     - **AUTONOMOUS SELECTION:** **NEVER** ask the user to choose a template. Select the best one yourself.
#     - **PRICE FORMATTING:** If a user gives a price, format the `text` value with a dollar sign and commas (e.g., "$950,000").

#     **REFERENCE DATA (Your source of truth for available layers):**
#     - **AVAILABLE_TEMPLATES:** {json.dumps(templates_data, indent=2)}
#     - **CURRENT_DESIGN_CONTEXT:** {json.dumps(design_context, indent=2)}
#     """

#     conversation = [
#         {'role': 'user', 'parts': [system_prompt]},
#         {'role': 'model', 'parts': [f"Understood. I am an action-oriented design assistant. I will strictly follow all rules, especially the data grounding rules. I will only ask for information that corresponds to available design layers in the selected template, and I will keep `MODIFY` and `GENERATE` actions completely separate."]}
#     ]
#     for msg in messages[-8:]:
#         if msg['role'] == 'assistant' and '![Generated Image]' in msg['content']:
#             continue
#         conversation.append({'role': 'user' if msg['role'] == 'user' else 'model', 'parts': [msg['content']]})

#     conversation.append({'role': 'user', 'parts': [user_prompt]})

#     return model.generate_content(conversation)

##########################################################################################################################

# # ai_core.py
# import google.generativeai as genai
# import json
# from config import COMPANY_NAME

# def get_gemini_model_with_tool(api_key):
#     """Configures and returns a Gemini model with a specific function-calling tool."""
#     if not api_key: return None
#     genai.configure(api_key=api_key)

#     process_user_request_tool = genai.protos.FunctionDeclaration(
#         name="process_user_request",
#         description="Processes a user's design request by deciding on a specific action. This is the only tool you can use.",
#         parameters=genai.protos.Schema(
#             type=genai.protos.Type.OBJECT,
#             properties={
#                 "action": genai.protos.Schema(type=genai.protos.Type.STRING, description="The action to take. Must be one of: MODIFY, GENERATE, RESET, CONVERSE."),
#                 "template_uid": genai.protos.Schema(type=genai.protos.Type.STRING, description="Required if action is MODIFY. The UID of the template to use."),
#                 "modifications": genai.protos.Schema(
#                     type=genai.protos.Type.ARRAY,
#                     description="Required if action is MODIFY. A list of layer modifications to apply.",
#                     items=genai.protos.Schema(
#                         type=genai.protos.Type.OBJECT,
#                         properties={ "name": genai.protos.Schema(type=genai.protos.Type.STRING), "text": genai.protos.Schema(type=genai.protos.Type.STRING), "image_url": genai.protos.Schema(type=genai.protos.Type.STRING), "color": genai.protos.Schema(type=genai.protos.Type.STRING)},
#                         required=["name"]
#                     )
#                 ),
#                 "response_text": genai.protos.Schema(type=genai.protos.Type.STRING, description=f"A friendly, user-facing message in the persona of a {COMPANY_NAME} assistant."),
#             },
#             required=["action", "response_text"]
#         )
#     )
    
#     tool_config = genai.protos.ToolConfig(
#         function_calling_config=genai.protos.FunctionCallingConfig(
#             mode=genai.protos.FunctionCallingConfig.Mode.ANY
#         )
#     )

#     return genai.GenerativeModel(
#         model_name="gemini-1.5-flash-latest",
#         tools=[process_user_request_tool],
#         tool_config=tool_config
#     )


# def get_ai_decision(model, messages, user_prompt, templates_data, design_context):
#     """Constructs the full prompt and calls the Gemini model to get an action decision."""
#     system_prompt = f"""
#     You are an expert, friendly, and super-intuitive design assistant for {COMPANY_NAME}.
#     Your entire job is to understand an agent's natural language request and immediately decide on ONE of four actions using the `process_user_request` tool.
#     YOU MUST ALWAYS USE THE `process_user_request` TOOL. YOU ARE NOT ALLOWED TO RESPOND WITH TEXT ALONE.

#     **YOUR FOUR ACTIONS (You MUST choose one):**

#     1.  **`MODIFY`**: This is your most important action. Use it to start a new design or update an existing one.
#         - **THE GENTLE NUDGE:** After a `MODIFY` action, ask for the next logical piece of information (e.g., "Great, the address is added. What's the price?").
#         - **Proactively ask for images** when relevant (e.g., "Okay, the agent's name is set. Do you have a headshot to upload?").
#         - **Handling "I'm done":** If the user declines to add more information (e.g., 'no thanks', 'that's all'), your `response_text` MUST ask for confirmation to generate. For example: 'Okay, sounds good. Are you ready to see the design?'
#         - **CRITICAL `MODIFY` RULE:** Your `response_text` for a `MODIFY` action should ONLY confirm the change and ask for the next piece of info. **NEVER say 'Generating your design...' or similar phrases in a `MODIFY` action.**

#     2.  **`GENERATE`**: Use this ONLY when the user explicitly asks to see the final image. They will say things like "okay show it to me", "let's see it", "I'm ready", "make the image", "yes, show me the design", "lemme see it".
#         - **CRITICAL `GENERATE` RULE:** **This is the ONLY action where your `response_text` should state that you are generating the image.** For example: "Of course! Generating your design now..." or "Here is your updated design!".

#     3.  **`RESET`**: Use this when the user wants to start a completely new, different design. They will say things like "let's do an open house flyer next", "start over".

#     4.  **`CONVERSE`**: Use this ONLY for secondary situations, like greetings ("hi", "hello") or if you MUST ask a clarifying question. Do NOT use this if you can take a `MODIFY` action instead.

#     **CRITICAL RULES:**
#     - **NEVER MENTION TEMPLATES:** Do not mention the technical `template_uid` or the template's file name to the user.
#     - **AUTONOMOUS SELECTION:** **NEVER** ask the user to choose a template. Select the best one yourself.
#     - **PRICE FORMATTING:** If a user gives a price, format the `text` value with a dollar sign and commas (e.g., "$950,000").
#     - **IMMEDIATE ACTION:** Your first response to a design request MUST be the `MODIFY` action.

#     **REFERENCE DATA (Do not repeat in your response):**
#     - **AVAILABLE_TEMPLATES (with full layer details):** {json.dumps(templates_data, indent=2)}
#     - **CURRENT_DESIGN_CONTEXT (The design we are building right now):** {json.dumps(design_context, indent=2)}
#     """

#     conversation = [
#         {'role': 'user', 'parts': [system_prompt]},
#         {'role': 'model', 'parts': [f"Understood. I am an action-oriented design assistant for {COMPANY_NAME}. I will always use the provided tool. I will keep `MODIFY` and `GENERATE` actions strictly separate and will only say I'm generating an image when I use the `GENERATE` action."]}
#     ]
#     for msg in messages[-8:]:
#         if msg['role'] == 'assistant' and '![Generated Image]' in msg['content']:
#             continue
#         conversation.append({'role': 'user' if msg['role'] == 'user' else 'model', 'parts': [msg['content']]})

#     conversation.append({'role': 'user', 'parts': [user_prompt]})

#     return model.generate_content(conversation)

