"""LLM backends. Each backend offers one or both of:

* ``label_logprobs(prompt, labels)`` -> (K,) log-probabilities of the option
  label tokens at the next position (log-softmax over the full vocabulary);
* ``sample(prompt, n, temperature, top_p, top_k, max_new_tokens)`` -> n completions.

``HFBackend`` (transformers) and ``LlamaCppBackend`` (GGUF files, e.g. quantized
models) support both; ``OpenAIBackend`` (any OpenAI-compatible server, including
vLLM) supports sampling only.

Prompts are sent as raw text, as in the paper. With ``chat_template=True`` the
HF backend wraps the prompt in the model's chat template with thinking disabled.
"""

from __future__ import annotations

import numpy as np


class HFBackend:
    def __init__(self, model: str, dtype: str = "bfloat16", device: str | None = None,
                 chat_template: bool = False, seed: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        if seed is not None:
            torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=getattr(torch, dtype), device_map=self.device, trust_remote_code=True)
        self.model.eval()
        self.chat_template = chat_template

    def _encode(self, prompt: str):
        if self.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def label_logprobs(self, prompt: str, labels: list[str]) -> np.ndarray:
        ids = [self.tokenizer.encode(lab, add_special_tokens=False)[0] for lab in labels]
        with self.torch.inference_mode():
            logits = self.model(**self._encode(prompt)).logits[0, -1, :]
            return self.torch.log_softmax(logits.float(), dim=-1)[ids].cpu().numpy()

    def sample(self, prompt: str, n: int, temperature: float, top_p: float, top_k: int,
               max_new_tokens: int) -> list[str]:
        inputs = self._encode(prompt)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs, do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p,
                max_new_tokens=max_new_tokens, num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id)
        new = out[:, inputs["input_ids"].shape[1]:]
        return [s.strip() for s in self.tokenizer.batch_decode(new, skip_special_tokens=True)]


class LlamaCppBackend:
    def __init__(self, model: str, n_ctx: int = 4096, n_gpu_layers: int = -1, seed: int | None = None):
        from llama_cpp import Llama

        kwargs = {"seed": seed} if seed is not None else {}
        self.llm = Llama(model_path=model, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                         logits_all=False, verbose=False, **kwargs)

    def label_logprobs(self, prompt: str, labels: list[str]) -> np.ndarray:
        ids = []
        for lab in labels:
            toks = self.llm.tokenize(lab.encode("utf-8"), add_bos=False, special=False)
            if len(toks) != 1:
                raise ValueError(f"label {lab!r} is {len(toks)} tokens; labels must be single tokens")
            ids.append(toks[0])
        self.llm.reset()
        self.llm.eval(self.llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True))
        x = np.ctypeslib.as_array(self.llm._ctx.get_logits(), shape=(self.llm.n_vocab(),))
        x = np.asarray(x, dtype=np.float32).copy()
        m = x.max()
        return (x - (m + np.log(np.exp(x - m).sum())))[ids]

    def sample(self, prompt: str, n: int, temperature: float, top_p: float, top_k: int,
               max_new_tokens: int) -> list[str]:
        return [
            self.llm.create_completion(prompt=prompt, max_tokens=max_new_tokens, temperature=temperature,
                                       top_k=top_k, top_p=top_p)["choices"][0]["text"].strip()
            for _ in range(n)
        ]


class OpenAIBackend:
    """Chat-completions endpoint; set OPENAI_API_KEY / OPENAI_BASE_URL or pass them."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 seed: int | None = None, timeout: float = 60.0, max_retries: int = 8):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model, self.seed = model, seed

    def label_logprobs(self, prompt: str, labels: list[str]) -> np.ndarray:
        raise NotImplementedError("the OpenAI backend supports mode='samples' only")

    def sample(self, prompt: str, n: int, temperature: float, top_p: float, top_k: int,
               max_new_tokens: int) -> list[str]:
        from openai import BadRequestError

        if top_k:
            raise ValueError("the OpenAI API has no top_k; set top_k=0")
        token_arg = ("max_completion_tokens" if self.model.startswith(("gpt-5", "o1", "o3", "o4"))
                     else "max_tokens")
        kwargs = {"seed": self.seed} if self.seed is not None else {}
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}], n=n,
                temperature=temperature, top_p=top_p, **{token_arg: max_new_tokens}, **kwargs)
        except BadRequestError as e:
            if getattr(e, "code", None) == "content_filter":   # blocked prompt: record empty answers
                return [""] * n
            raise
        return [(c.message.content or "").strip() for c in resp.choices]


BACKENDS = {"hf": HFBackend, "llamacpp": LlamaCppBackend, "openai": OpenAIBackend}
