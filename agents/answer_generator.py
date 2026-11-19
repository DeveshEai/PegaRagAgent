"""
Answer generation, streaming, completeness guards, model resolution/fallback,
and direct grounded fallback for PegaRagAgent.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from groq import Groq

from config.settings import (
    DEFAULT_FINAL_EVIDENCE_K,
    FALLBACK_GROQ_MODEL,
    PRIMARY_GROQ_MODEL,
    get_model_spec,
)
from retrieval.grounding import (
    build_grounded_context_prompt,
    contextualize_query_with_history,
    deduplicate_sources,
    estimate_tokens,
    verify_citations,
)


def is_context_length_error(err_msg: str) -> bool:
    """Checks if a Groq error message indicates a context_length_exceeded condition."""
    low = str(err_msg).lower()
    return any(x in low for x in [
        "context_length_exceeded",
        "reduce the length of the messages",
        "context length",
        "maximum context length"
    ])


GROUNDED_SYSTEM_PROMPT = (
    "You are a Senior Pega Certified Lead System Architect (CLSA) advising a fellow architect in a natural, conversational peer-to-peer style.\n\n"
    "RESPONSE STYLE & COMPLEXITY:\n"
    "1. Answer directly from the first sentence without mandatory or formulaic headings (never use boilerplate headers such as 'Short answer', 'What the excerpts actually tell us', 'Implication for architecture and governance', or repetitive 'Bottom line' wrap-ups).\n"
    "2. Match length to question complexity:\n"
    "   - Simple definitions (e.g., 'What is X?'): 80–150 words.\n"
    "   - Technical explanations or comparisons: 150–300 words.\n"
    "   - Complex architecture scenarios: as detailed as necessary.\n"
    "3. Include practical Pega examples (e.g., `pyWorkPage`, `Queue Processor`, `Job Scheduler`, `Rule-Obj-Class`, `Work-`, `Data-`, `Int-`) and explain architectural implications (performance, locking, reusability, governance) when genuinely relevant.\n\n"
    "STRICT GROUNDING & CITATION INTEGRITY:\n"
    "4. Clearly separate retrieved evidence from general Pega knowledge: cite `[filename, Page X, Section: Y]` only for claims directly supported by the provided excerpts.\n"
    "5. Never present general Pega knowledge as if it came from the retrieved documents, and never invent citations or attach unrelated sources to a claim.\n"
    "6. If the retrieved excerpts lack sufficient detail, explicitly state that gap (for example: 'The retrieved curriculum excerpts do not cover ...') and still provide helpful general Pega CLSA guidance clearly marked as general knowledge without citations."
)


def find_best_alternative_model(client: Groq, current_model: str) -> Optional[str]:
    """Dynamically finds an active text chat model from the user's specific Groq account."""
    try:
        live_models = client.models.list()
        raw_ids = [m.id for m in live_models.data]
        valid_ids = []
        for m_id in raw_ids:
            m_low = m_id.lower()
            if any(x in m_low for x in ["whisper", "orpheus", "tts", "audio", "guard", "safeguard", "canopylabs"]):
                continue
            if m_id != current_model:
                valid_ids.append(m_id)

        priority = [
            FALLBACK_GROQ_MODEL,
            PRIMARY_GROQ_MODEL,
            "qwen/qwen3.8-27b",
            "allam-2-7b"
        ]
        for p in priority:
            if p in valid_ids:
                return p
        return valid_ids[0] if valid_ids else None
    except Exception:
        return None


def resolve_available_model(client: Groq, requested_model: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Validates the requested model against Groq's live Models API.
    If unavailable or retired, automatically falls back to openai/gpt-oss-20b
    (or the best active text model on the account).
    """
    target = (requested_model or "").strip() or PRIMARY_GROQ_MODEL
    try:
        live_models = client.models.list()
        active_ids = [
            m.id for m in live_models.data
            if not any(x in m.id.lower() for x in ["whisper", "orpheus", "tts", "audio", "guard", "safeguard", "canopylabs"])
        ]
        if not active_ids:
            return target, None
        if target in active_ids:
            return target, None
        if FALLBACK_GROQ_MODEL in active_ids:
            return FALLBACK_GROQ_MODEL, f"Requested model `{target}` is unavailable on Groq; using fallback `{FALLBACK_GROQ_MODEL}`."
        if PRIMARY_GROQ_MODEL in active_ids:
            return PRIMARY_GROQ_MODEL, f"Requested model `{target}` is unavailable on Groq; using `{PRIMARY_GROQ_MODEL}`."
        alt = active_ids[0]
        return alt, f"Requested model `{target}` is unavailable on Groq; using `{alt}`."
    except Exception:
        return target, None


def trim_to_last_complete_sentence(text: str) -> str:
    """Trims a length-truncated response cleanly to the last complete sentence boundary."""
    if not text:
        return ""
    stripped = text.rstrip()
    last_end = max(stripped.rfind("."), stripped.rfind("!"), stripped.rfind("?"))
    if last_end != -1 and last_end > len(stripped) * 0.4:
        return stripped[:last_end + 1]
    last_nl = stripped.rfind("\n")
    if last_nl != -1 and last_nl > len(stripped) * 0.4:
        return stripped[:last_nl]
    return stripped


def create_completion_with_completeness_guard(
    client: Groq,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int
) -> Tuple[str, Optional[str]]:
    """
    Executes a Groq chat completion and inspects finish_reason.
    If finish_reason == 'length', performs at most ONE controlled retry with a concise
    completion instruction and expanded token budget (within model context_window).
    If still truncated, trims cleanly at the last complete sentence and appends a notice.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    choice = response.choices[0]
    content = choice.message.content or ""
    finish_reason = getattr(choice, "finish_reason", None)

    if finish_reason != "length":
        return content, finish_reason

    spec = get_model_spec(model_name)
    ctx_win = spec.get("context_window", 4096)
    prompt_tokens_est = sum(estimate_tokens(m.get("content", "")) for m in messages)
    headroom = ctx_win - prompt_tokens_est - 256
    expanded_tokens = min(2200, max(max_tokens + 400, int(max_tokens * 1.5)), headroom)

    if expanded_tokens > max_tokens + 100:
        retry_messages = list(messages[:-1]) + [{
            "role": messages[-1]["role"],
            "content": messages[-1]["content"] + "\n\n(Be concise and complete your response cleanly within the token limit.)"
        }]
        try:
            retry_resp = client.chat.completions.create(
                model=model_name,
                messages=retry_messages,
                temperature=temperature,
                max_tokens=expanded_tokens
            )
            retry_choice = retry_resp.choices[0]
            retry_content = retry_choice.message.content or ""
            retry_reason = getattr(retry_choice, "finish_reason", None)
            if retry_reason != "length" and retry_content:
                return retry_content, retry_reason
            if retry_content:
                content = retry_content
        except Exception:
            pass

    trimmed = trim_to_last_complete_sentence(content)
    notice = "\n\n> *Note: Response reached the model's output length limit and was trimmed at the last complete sentence.*"
    return (trimmed + notice) if trimmed else "No complete response generated.", "length"


def stream_completion_with_completeness_guard(
    client: Groq,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int
):
    """
    Streams tokens from Groq (stream=True) and inspects final finish_reason.
    Yields:
      - {"type": "token", "text": str}
      - {"type": "complete", "text": str, "finish_reason": Optional[str]}
    """
    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )
    accumulated = []
    finish_reason = None

    try:
        for chunk in stream:
            if not chunk or not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            delta_text = getattr(delta, "content", None) if delta else None
            if delta_text:
                accumulated.append(delta_text)
                yield {"type": "token", "text": delta_text}
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
    except Exception as stream_err:
        partial = "".join(accumulated).strip()
        if not partial:
            raise stream_err
        trimmed = trim_to_last_complete_sentence(partial)
        yield {
            "type": "complete",
            "text": trimmed + "\n\n> *Note: Stream was interrupted and trimmed at the last complete sentence.*",
            "finish_reason": "interrupted"
        }
        return

    full_text = "".join(accumulated)
    if finish_reason != "length":
        yield {"type": "complete", "text": full_text, "finish_reason": finish_reason}
        return

    spec = get_model_spec(model_name)
    ctx_win = spec.get("context_window", 4096)
    prompt_tokens_est = sum(estimate_tokens(m.get("content", "")) for m in messages)
    headroom = ctx_win - prompt_tokens_est - 256
    expanded_tokens = min(2200, max(max_tokens + 400, int(max_tokens * 1.5)), headroom)

    if expanded_tokens > max_tokens + 100:
        retry_messages = list(messages[:-1]) + [{
            "role": messages[-1]["role"],
            "content": messages[-1]["content"] + "\n\n(Be concise and complete your response cleanly within the token limit.)"
        }]
        try:
            retry_text, retry_reason = create_completion_with_completeness_guard(
                client=client,
                model_name=model_name,
                messages=retry_messages,
                temperature=temperature,
                max_tokens=expanded_tokens
            )
            if retry_text:
                yield {"type": "complete", "text": retry_text, "finish_reason": retry_reason}
                return
        except Exception:
            pass

    trimmed = trim_to_last_complete_sentence(full_text)
    notice = "\n\n> *Note: Response reached the model's output length limit and was trimmed at the last complete sentence.*"
    yield {
        "type": "complete",
        "text": (trimmed + notice) if trimmed else "No complete response generated.",
        "finish_reason": "length"
    }


def execute_direct_grounded_fallback(
    engine: Any,
    query: str,
    client: Groq,
    model_name: str,
    temperature: float = 0.0,
    chat_history: Optional[List[Dict[str, str]]] = None,
    execution_trace: Optional[List[Dict[str, Any]]] = None,
    reason: str = "",
    top_k: int = DEFAULT_FINAL_EVIDENCE_K,
    use_reranker: bool = True,
    dense_weight: float = 0.5,
    section_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Direct Grounded Retrieval Fallback when a model does not support Groq function/tool calling
    or when a straightforward query is routed directly.
    """
    if execution_trace is None:
        execution_trace = []

    spec = get_model_spec(model_name)
    prompt_budget = spec.get("prompt_budget", 2200)
    max_completion_tokens = spec.get("max_completion_tokens", 600)

    search_query = contextualize_query_with_history(query, chat_history)
    retrieved_chunks = engine.search(
        query=search_query,
        top_k=top_k,
        section_filter=section_filter,
        use_reranker=use_reranker,
        dense_weight=dense_weight
    )

    query_lower = search_query.lower()
    if any(w in query_lower for w in ["how to", "configure", "implement", "step", "challenge", "exercise", "create", "setup"]):
        try:
            challenge_res = engine.tool_search_challenges(query=search_query, top_k=2)
            if "results" in challenge_res:
                for r in challenge_res["results"]:
                    if not any(c.get("document") == r.get("excerpt") for c in retrieved_chunks):
                        retrieved_chunks.append({
                            "source": r.get("document", "Challenge"),
                            "page": str(r.get("page", "1")),
                            "section": r.get("section", "Hands-on Challenges"),
                            "category": "Hands-on Challenge",
                            "score": float(r.get("relevance_score", "0%").replace("%", "")),
                            "document": r.get("excerpt", "")
                        })
        except Exception:
            pass

    context_str, retained_sources = build_grounded_context_prompt(
        retrieved_chunks, max_token_budget=prompt_budget
    )

    execution_trace.append({
        "step": len(execution_trace) + 1,
        "tool": "direct_grounded_retrieval",
        "args": {
            "model": model_name,
            "sources_retrieved": len(retained_sources),
            "prompt_budget": prompt_budget,
            "reason": reason or f"Model '{model_name}' operates with direct grounded retrieval."
        },
        "summary": f"Retrieved {len(retained_sources)} top-ranked Pega curriculum excerpts via Hybrid Search (Dense + BM25 RRF + Reranker)."
    })

    system_prompt = GROUNDED_SYSTEM_PROMPT
    user_content = (
        f"USER QUESTION:\n{query}\n\n"
        f"OFFICIAL PEGA ACADEMY CONTEXT EXCERPTS:\n"
        f"{context_str}\n\n"
        f"Answer clearly and naturally using only the verified excerpts above, with compact inline citations `[filename, Page X, Section: Y]`:"
    )

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        ctx_win = spec.get("context_window", 4096)
        hist_limit = 6 if ctx_win >= 16384 else 2
        char_limit = 700 if ctx_win >= 16384 else 350
        for h in chat_history[-hist_limit:]:
            role = h.get("role", "user")
            if role in ("user", "assistant") and h.get("content"):
                messages.append({"role": role, "content": h["content"][:char_limit]})
    messages.append({"role": "user", "content": user_content})

    try:
        raw_answer, _ = create_completion_with_completeness_guard(
            client=client,
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        final_sources = deduplicate_sources(retained_sources)
        verified_answer, _ = verify_citations(raw_answer or "No response generated.", final_sources)
        return {
            "answer": verified_answer,
            "trace": execution_trace,
            "sources": final_sources
        }
    except Exception as e:
        err_str = str(e)
        if is_context_length_error(err_str):
            reduced_budget = max(800, prompt_budget // 2)
            reduced_completion = max(300, max_completion_tokens // 2)
            reduced_context_str, reduced_sources = build_grounded_context_prompt(
                retrieved_chunks, max_token_budget=reduced_budget
            )
            retry_user_content = (
                f"USER QUESTION:\n{query}\n\n"
                f"OFFICIAL PEGA ACADEMY CONTEXT EXCERPTS:\n"
                f"{reduced_context_str}\n\n"
                f"Answer clearly and naturally using only the verified excerpts above, with compact inline citations `[filename, Page X, Section: Y]`:"
            )
            retry_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": retry_user_content}
            ]
            execution_trace.append({
                "step": len(execution_trace) + 1,
                "tool": "context_recovery_retry",
                "args": {
                    "model": model_name,
                    "reduced_budget": reduced_budget,
                    "reduced_completion": reduced_completion
                },
                "summary": f"Reduced context budget to {reduced_budget} tokens to fit model limits."
            })
            try:
                retry_res = client.chat.completions.create(
                    model=model_name,
                    messages=retry_messages,
                    temperature=temperature,
                    max_tokens=reduced_completion
                )
                final_reduced = deduplicate_sources(reduced_sources)
                raw_retry = retry_res.choices[0].message.content or "No response generated."
                verified_retry, _ = verify_citations(raw_retry, final_reduced)
                return {
                    "answer": verified_retry,
                    "trace": execution_trace,
                    "sources": final_reduced
                }
            except Exception:
                return {
                    "answer": f"⚠️ **Context Length Notice:** The query and evidence exceed the maximum token capacity ({spec.get('context_window', 4096)} tokens) for model `{model_name}`. Please switch to a higher-capacity model like `openai/gpt-oss-120b` in Configuration.",
                    "trace": execution_trace,
                    "sources": deduplicate_sources(reduced_sources)
                }

        err_low = err_str.lower()
        is_model_issue = any(x in err_low for x in [
            "model_not_found", "model_terms_required", "model_decommissioned",
            "decommissioned", "deprecated", "does not exist", "not have access"
        ])
        if is_model_issue:
            alt_model = find_best_alternative_model(client, model_name)
            if alt_model and alt_model != model_name:
                alt_spec = get_model_spec(alt_model)
                alt_tokens = alt_spec.get("max_completion_tokens", 1000)
                try:
                    response = client.chat.completions.create(
                        model=alt_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=alt_tokens
                    )
                    final_sources = deduplicate_sources(retained_sources)
                    raw_alt = response.choices[0].message.content or ""
                    verified_alt, _ = verify_citations(raw_alt, final_sources)
                    return {
                        "answer": f"> *Note: Requested model `{model_name}` was unavailable. Auto-switched to `{alt_model}`.*\n\n" + verified_alt,
                        "trace": execution_trace,
                        "sources": final_sources
                    }
                except Exception as retry_err:
                    return {
                        "answer": f"❌ **Groq API Error:** {str(retry_err)}",
                        "trace": execution_trace,
                        "sources": deduplicate_sources(retained_sources)
                    }
        return {
            "answer": f"❌ **Groq API Error:** {str(e)}",
            "trace": execution_trace,
            "sources": deduplicate_sources(retained_sources)
        }


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    groq_api_key: str,
    model_name: str = PRIMARY_GROQ_MODEL,
    temperature: float = 0.2,
    chat_history: Optional[List[Dict[str, str]]] = None,
    groq_factory: Callable[..., Any] = Groq
) -> str:
    """Sends context, conversation history, and question to Groq LLM for a grounded response with citations."""
    if not groq_api_key or not groq_api_key.strip():
        return (
            "⚠️ **Groq API Key Required:** Please provide your Groq API key in the sidebar to generate answers. "
            "You can get a free key instantly at [https://console.groq.com/keys](https://console.groq.com/keys)."
        )

    client = groq_factory(api_key=groq_api_key.strip())
    model_name, _ = resolve_available_model(client, model_name)

    spec = get_model_spec(model_name)
    prompt_budget = spec.get("prompt_budget", 4500)
    max_completion_tokens = spec.get("max_completion_tokens", 1200)

    context_str, retained_sources = build_grounded_context_prompt(
        retrieved_chunks, max_token_budget=prompt_budget
    )
    system_prompt = GROUNDED_SYSTEM_PROMPT
    user_prompt = (
        f"USER QUESTION:\n{query}\n\n"
        f"RELEVANT PEGA ACADEMY CONTEXT EXCERPTS:\n"
        f"{context_str}\n\n"
        f"Answer clearly and naturally using only the verified excerpts above, with compact inline citations `[filename, Page X, Section: Y]`:"
    )

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        ctx_win = spec.get("context_window", 4096)
        hist_limit = 6 if ctx_win >= 16384 else 2
        char_limit = 700 if ctx_win >= 16384 else 350
        for h in chat_history[-hist_limit:]:
            role = h.get("role", "user")
            if role in ("user", "assistant") and h.get("content"):
                messages.append({"role": role, "content": h["content"][:char_limit]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        try:
            raw_ans, _ = create_completion_with_completeness_guard(
                client=client,
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_completion_tokens
            )
            verified_ans, _ = verify_citations(raw_ans, retained_sources)
            return verified_ans
        except Exception as model_err:
            err_str = str(model_err)
            if is_context_length_error(err_str):
                reduced_budget = max(800, prompt_budget // 2)
                reduced_completion = max(300, max_completion_tokens // 2)
                reduced_context_str, reduced_sources = build_grounded_context_prompt(
                    retrieved_chunks, max_token_budget=reduced_budget
                )
                retry_user_prompt = (
                    f"USER QUESTION:\n{query}\n\n"
                    f"RELEVANT PEGA ACADEMY CONTEXT EXCERPTS:\n"
                    f"{reduced_context_str}\n\n"
                    f"Answer clearly and naturally using only the verified excerpts above, with compact inline citations `[filename, Page X, Section: Y]`:"
                )
                try:
                    retry_response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": retry_user_prompt}
                        ],
                        temperature=temperature,
                        max_tokens=reduced_completion
                    )
                    retry_choice = retry_response.choices[0]
                    raw_retry = retry_choice.message.content or ""
                    if getattr(retry_choice, "finish_reason", None) == "length":
                        raw_retry = trim_to_last_complete_sentence(raw_retry) + "\n\n> *Note: Response reached the model's output length limit and was trimmed at the last complete sentence.*"
                    verified_retry, _ = verify_citations(raw_retry, reduced_sources)
                    return verified_retry
                except Exception:
                    return f"⚠️ **Context Length Notice:** The query and evidence exceed the maximum token capacity for model `{model_name}`. Please switch to a higher-capacity model like `openai/gpt-oss-120b` in Configuration."

            err_lower = err_str.lower()
            is_model_err = any(x in err_lower for x in [
                "model_not_found", "model_terms_required", "model_decommissioned",
                "decommissioned", "deprecated", "does not exist", "not have access"
            ])
            if is_model_err:
                alt_model = find_best_alternative_model(client, model_name)
                if alt_model and alt_model != model_name:
                    alt_spec = get_model_spec(alt_model)
                    alt_tokens = alt_spec.get("max_completion_tokens", 1000)
                    raw_alt, _ = create_completion_with_completeness_guard(
                        client=client,
                        model_name=alt_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=alt_tokens
                    )
                    verified_alt, _ = verify_citations(raw_alt, retained_sources)
                    return f"> *Note: Requested model `{model_name}` was unavailable. Auto-switched to `{alt_model}`.*\n\n" + verified_alt
            raise model_err
    except Exception as e:
        return f"❌ **Error generating response from Groq:** {str(e)}"


def stream_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    groq_api_key: str,
    model_name: str = PRIMARY_GROQ_MODEL,
    temperature: float = 0.0,
    chat_history: Optional[List[Dict[str, str]]] = None,
    groq_factory: Callable[..., Any] = Groq
):
    """
    Streams a grounded answer progressively while preserving token budgeting,
    context-length recovery, finish_reason completeness guard, and citation verification.
    """
    if not groq_api_key or not groq_api_key.strip():
        msg = (
            "⚠️ **Groq API Key Required:** Please provide your Groq API key in Configuration to generate answers. "
            "You can get a free key at [console.groq.com/keys](https://console.groq.com/keys)."
        )
        yield {"type": "final", "answer": msg, "sources": [], "model_used": model_name or PRIMARY_GROQ_MODEL}
        return

    client = groq_factory(api_key=groq_api_key.strip())
    model_name, model_notice = resolve_available_model(client, model_name)

    spec = get_model_spec(model_name)
    prompt_budget = spec.get("prompt_budget", 4500)
    max_completion_tokens = spec.get("max_completion_tokens", 1200)

    context_str, retained_sources = build_grounded_context_prompt(
        retrieved_chunks, max_token_budget=prompt_budget
    )
    final_sources = deduplicate_sources(retained_sources)

    user_prompt = (
        f"USER QUESTION:\n{query}\n\n"
        f"RELEVANT PEGA ACADEMY CONTEXT EXCERPTS:\n"
        f"{context_str}\n\n"
        f"Answer clearly and naturally using only the verified excerpts above, with compact inline citations `[filename, Page X, Section: Y]`:"
    )

    messages = [{"role": "system", "content": GROUNDED_SYSTEM_PROMPT}]
    if chat_history:
        ctx_win = spec.get("context_window", 4096)
        hist_limit = 6 if ctx_win >= 16384 else 2
        char_limit = 700 if ctx_win >= 16384 else 350
        for h in chat_history[-hist_limit:]:
            role = h.get("role", "user")
            if role in ("user", "assistant") and h.get("content"):
                messages.append({"role": role, "content": h["content"][:char_limit]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        try:
            completed_text = ""
            for ev in stream_completion_with_completeness_guard(
                client=client,
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_completion_tokens
            ):
                if ev["type"] == "token":
                    yield ev
                elif ev["type"] == "complete":
                    completed_text = ev["text"]
            verified_ans, _ = verify_citations(completed_text, final_sources)
            prefix_note = f"> *Note: {model_notice}*\n\n" if model_notice else ""
            yield {"type": "final", "answer": prefix_note + verified_ans, "sources": final_sources, "model_used": model_name}
            return
        except Exception as model_err:
            err_str = str(model_err)
            if is_context_length_error(err_str):
                reduced_budget = max(800, prompt_budget // 2)
                reduced_completion = max(300, max_completion_tokens // 2)
                reduced_context_str, reduced_sources = build_grounded_context_prompt(
                    retrieved_chunks, max_token_budget=reduced_budget
                )
                retry_user_prompt = (
                    f"USER QUESTION:\n{query}\n\n"
                    f"RELEVANT PEGA ACADEMY CONTEXT EXCERPTS:\n"
                    f"{reduced_context_str}\n\n"
                    f"Answer clearly and naturally using only the verified excerpts above, with compact inline citations `[filename, Page X, Section: Y]`:"
                )
                retry_messages = [
                    {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                    {"role": "user", "content": retry_user_prompt}
                ]
                completed_retry = ""
                for ev in stream_completion_with_completeness_guard(
                    client=client,
                    model_name=model_name,
                    messages=retry_messages,
                    temperature=temperature,
                    max_tokens=reduced_completion
                ):
                    if ev["type"] == "token":
                        yield ev
                    elif ev["type"] == "complete":
                        completed_retry = ev["text"]
                final_reduced = deduplicate_sources(reduced_sources)
                verified_retry, _ = verify_citations(completed_retry, final_reduced)
                yield {"type": "final", "answer": verified_retry, "sources": final_reduced}
                return

            err_lower = err_str.lower()
            is_model_err = any(x in err_lower for x in [
                "model_not_found", "model_terms_required", "model_decommissioned",
                "decommissioned", "deprecated", "does not exist", "not have access"
            ])
            if is_model_err:
                alt_model = find_best_alternative_model(client, model_name)
                if alt_model and alt_model != model_name:
                    alt_spec = get_model_spec(alt_model)
                    alt_tokens = alt_spec.get("max_completion_tokens", 1000)
                    completed_alt = ""
                    for ev in stream_completion_with_completeness_guard(
                        client=client,
                        model_name=alt_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=alt_tokens
                    ):
                        if ev["type"] == "token":
                            yield ev
                        elif ev["type"] == "complete":
                            completed_alt = ev["text"]
                    verified_alt, _ = verify_citations(completed_alt, final_sources)
                    note = f"> *Note: Requested model `{model_name}` was unavailable. Auto-switched to `{alt_model}`.*\n\n"
                    yield {"type": "final", "answer": note + verified_alt, "sources": final_sources, "model_used": alt_model}
                    return
            raise model_err
    except Exception as e:
        yield {
            "type": "final",
            "answer": f"❌ **Error generating response from Groq:** {str(e)}",
            "sources": final_sources,
            "model_used": model_name
        }
