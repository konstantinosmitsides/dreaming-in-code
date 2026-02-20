import asyncio
import os
from typing import Any, Literal

from openai import AsyncOpenAI

class LLM:
	def __init__(
		self,
		provider: str,
		base_url: str,
		model: str,
		llm_type: Literal["generation", "embedding"],
		max_tokens: int = None,
		temperature: float = None,
		top_p: float = None,
		think: bool = False,
		embedding_size: int = 1024,
	):
		self.provider = provider
		self.base_url = base_url
		self.model = model
		self.llm_type = llm_type
		self.embedding_size = embedding_size

		# parameters for generation
		if self.llm_type == "generation":
			self.max_tokens = max_tokens
			self.temperature = temperature
			self.top_p = top_p
			self.think = think
		elif self.llm_type == "embedding":
			self.max_tokens = None
			self.temperature = None
			self.top_p = None
			self.think = False

		self.client = self._create_client()

	def _create_client(self):
		if self.provider == "local":
			return AsyncOpenAI(base_url=self.base_url, api_key="token-")
		elif self.provider == "gemini":
			api_key = os.getenv("GEMINI_API_KEY")
			return AsyncOpenAI(base_url=self.base_url, api_key=api_key)
		elif self.provider == "openai":
			api_key = os.getenv("OPENAI_API_KEY")
			return AsyncOpenAI(api_key=api_key)
		elif self.provider == "openrouter":
			api_key = os.getenv("OPENROUTER_API_KEY")
			# Default OpenRouter Base URL if not provided
			base_url = self.base_url or "https://openrouter.ai/api/v1"
			return AsyncOpenAI(
				base_url=base_url,
				api_key=api_key,
			)
		else:
			raise ValueError(f"Provider {self.provider} not supported")

	async def _query_local_gen(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
		if self.think:
			messages = [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			]
		else:
			messages = [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt + " /no_think"},
			]

		try:
			chat_completion = await self.client.chat.completions.create(
				model=self.model,
				messages=messages,
				max_tokens=self.max_tokens,
				temperature=self.temperature,
				top_p=self.top_p,
			)

			return {
				"system_prompt": system_prompt,
				"user_prompt": user_prompt,
				"content": chat_completion.choices[0].message.content,
				"reasoning_content": getattr(chat_completion.choices[0].message, "reasoning_content", None),
				"error": None,
			}
		except Exception as e:
			return {
				"system_prompt": system_prompt,
				"user_prompt": user_prompt,
				"content": None,
				"reasoning_content": None,
				"error": e,
			}

	async def _query_with_retries(self, api_call_coroutine, max_retries=3, initial_delay=2):
		"""A wrapper to add exponential backoff retries to an API call."""
		for attempt in range(max_retries):
			try:
				# Await the actual API call coroutine
				result = await api_call_coroutine

				# Check for a valid content response
				if result.get("content") is not None and result.get("content").strip():
					return result  # Success

				# Handle cases where the API returns a valid response but empty content
				error_message = (
					f"LLM returned empty content. Attempt {attempt + 1} of {max_retries}."
				)
				print(f"Warning: {error_message}")
				result["error"] = ValueError(error_message)

			except Exception as e:
				# Handle network errors, API errors, etc.
				print(
					f"Warning: LLM API call failed with error: {e}. Attempt {attempt + 1} of {max_retries}."
				)
				result = {"content": None, "error": e}

			# If we're not on the last attempt, wait before retrying
			if attempt < max_retries - 1:
				await asyncio.sleep(initial_delay * (2**attempt))  # Exponential backoff
			else:
				print(f"Error: LLM call failed after {max_retries} retries.")
				return result  # Return the last failed result

	async def _query_local_embed(
		self, texts_to_embed: str | list[str] | tuple[str, ...], instruction: str = None
	) -> dict[str, Any]:
		if isinstance(texts_to_embed, str):
			input_list = [texts_to_embed]
			return_single_result = True
		elif isinstance(texts_to_embed, (list, tuple)):
			input_list = list(texts_to_embed)
			return_single_result = False
		else:
			raise ValueError(f"Invalid input type: {type(texts_to_embed)}")

		# --- Start of new code for instruction ---
		if instruction:
			formatted_input_list = [
				f"Instruct: {instruction}\nQuery: {text}" for text in input_list
			]
		else:
			formatted_input_list = input_list
		# --- End of new code for instruction ---

		if not formatted_input_list:
			return []
		try:
			response = await self.client.embeddings.create(
				model=self.model,
				input=formatted_input_list,  # Updated to use the formatted list
			)
			results = []
			for i, result in enumerate(response.data):
				sliced_embedding = result.embedding[: self.embedding_size]
				results.append(
					{
						"input_text": input_list[i],  # Use the original text for clarity
						"embedding": sliced_embedding,
						"embedding_dim": len(sliced_embedding),
						"error": None,
					}
				)
			return results
		except Exception as e:
			return [
				{
					"input_text": text,
					"embedding": None,
					"embedding_dim": 0,
					"error": str(e),
				}
				for text in input_list
			]

	async def _query_gemini(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		]

		try:
			chat_completion = await self.client.chat.completions.create(
				model=self.model,
				messages=messages,
				reasoning_effort="high",
			)

			return {
				"system_prompt": system_prompt,
				"user_prompt": user_prompt,
				"content": chat_completion.choices[0].message.content,
				"error": None,
			}
		except Exception as e:
			return {
				"system_prompt": system_prompt,
				"user_prompt": user_prompt,
				"content": None,
				"error": e,
			}

	async def _query_batch_local_gen(
		self, system_prompt: str, user_prompts: list[str]
	) -> list[dict[str, Any]]:
		tasks = [self._query_local_gen(system_prompt, prompt) for prompt in user_prompts]
		results = await asyncio.gather(*tasks)
		return results

	async def _query_batch_gemini(
		self, system_prompt: str, user_prompts: list[str]
	) -> list[dict[str, Any]]:
		tasks = [self._query_gemini(system_prompt, prompt) for prompt in user_prompts]
		results = await asyncio.gather(*tasks)
		return results

	def query(self, system_prompt: str, user_prompts: list[str]) -> list[dict[str, Any]]:
		if self.provider == "local":
			return asyncio.run(self._query_batch_local_gen(system_prompt, user_prompts))
		elif self.provider == "gemini":
			return asyncio.run(self._query_batch_gemini(system_prompt, user_prompts))
		else:
			raise ValueError(f"Provider {self.provider} not supported")

	def get_embedding(
		self, text_to_embed: str | list[str] | tuple[str, ...], instruction: str = None
	) -> dict[str, Any]:
		if self.provider == "local" or self.provider == "openai" or self.provider == "openrouter":
			return asyncio.run(self._query_local_embed(text_to_embed, instruction))
		elif self.provider == "gemini":
			raise NotImplementedError("Gemini embeddings not yet implemented")
		else:
			raise ValueError(f"Provider {self.provider} not supported")
