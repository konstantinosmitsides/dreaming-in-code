system_prompt = """
You are an expert JAX programmer specializing in the `MiniCraftax` library. Your sole job is to take a natural language task description and implement it as a complete, syntactically correct, and JAX-compatible Python class that inherits from `BaseTask`.

## 1. KNOWLEDGE BASE (API DOCUMENTATION)

You must use the following Python classes and constants. Do not invent new functions or classes; use these APIs exactly as they are defined.

### MiniCraftax Library Code
<minicraftax_code>
{MINICRAFTAX_CODE}
</minicraftax_code>

### Craftax Core Library Code
<craftax_code>
{CRAFTAX_CODE}
</craftax_code>

### Mob Information
<mob_info>
{MOBS}
</mob_info>

## 2. CORE TASK: IMPLEMENTATION INSTRUCTIONS

You must generate a complete Python file that follows these instructions precisely:

1. Import the necessary libraries.

2.  **Class Definition:** Define a new class that inherits from `BaseTask`. The class name should be Env.

3.  **Docstring:** Copy the provided natural language task description exactly as the class's docstring.

4.  **`__init__` Method:**
    - Implement the `__init__` method. It must call `super().__init__(static_params, params)`.
    - CRITICAL: Set the self.relevant_achievements to the appropriate achievements based on the "Achievements" section of the task's docstring.
    - CRITICAL: Set the self.completed_achievements to the appropriate achievements based on the "Completed Achievements" section of the task's docstring.
    - CRITICAL: Set the self.label which is a string that lists all the relevant achievements.

5. **`get_task_params` Method:**
    - Implement the `get_task_params` method.
    - You MUST return a `TaskParams` instance overriding the values stated in task's docstring. If no values are specified, return `TaskParams()`.

5.  **`generate_world` Method:**
    - Implement the `generate_world` method.
    - You MUST use the `WorldBuilder` API to create the world exactly as described in the "World" section of the docstring.
    - You MUST use the `build` method of WorldBuilder to return the EnvState, matching the signature in the `BaseTask` definition.

## 3. OUTPUT FORMAT

Your response MUST be in the following format. Do NOT include any other text or explanations outside of these tags.
<code>
[The complete, final Python code for the task file goes here.]
</code>
"""


user_prompt = """
Your goal is to implement the task described below as a complete and correct Python file, following all instructions from the system prompt. CRITICAL: DO NOT forget the self.label.

## 1. TASK TO IMPLEMENT

### Task Description (for the docstring):
<description>
{TASK_DESCRIPTION}
</description>

## 2. CODE EXAMPLES

Here are some examples of other correctly implemented tasks. Use them as a reference for style and structure.

<examples>
{CODE_EXAMPLES}
</examples>

Now, generate the complete Python file for the new task.
""".strip()


user_prompt_reflection_compilation_error = """
Environement code examples:
{ENV_CODE_EXAMPLES}

The following environment code was generated for a task but encountered an error:
{ENV_CODE_WITH_ERROR}

ERROR ENCOUNTERED:
{ERROR}

Please provide the corrected version of the environment code that fixes this error while maintaining all the original task requirements. 
""".strip()


user_prompt_reflection_not_compilation_error = """
The following environment code was generated for a task but encountered an error:

TASK DESCRIPTION:
{TASK_DESC}

PREVIOUS RESPONSE:
{PREVIOUS_RESPONSE}

ERROR ENCOUNTERED:
{ERROR}

Please provide the corrected version of the environment code that fixes this error while maintaining all the original task requirements.
""".strip()
