"""
Autonomous research agent orchestration (non-streaming and streaming)
for PegaRagAgent.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from groq import Groq

from agents.answer_generator import (
    execute_direct_grounded_fallback,
    find_best_alternative_model,
    generate_answer,
    is_context_length_error,
    resolve_available_model,
    stream_answer,
    trim_to_last_complete_sentence,
)
from agents.tools import collect_tool_sources, get_agent_tools_schema
from config.settings import (
    DEFAULT_FINAL_EVIDENCE_K,
    PRIMARY_GROQ_MODEL,
    get_model_spec,
    is_tool_capable_model,
)
from retrieval.grounding import (
    contextualize_query_with_history,
    deduplicate_sources,
    verify_citations,
)

AGENT_SYSTEM_PROMPT = (
    "You are a Senior Pega Certified Lead System Architect (CLSA) advising a fellow architect in a natural, conversational peer-to-peer style.\n\n"
    "TOOL USE & RESEARCH:\n"
    "1. Call your tools (`search_curriculum`, `search_challenges`, `read_full_page`, `lookup_pega_terms`) to retrieve verified Pega excerpts before answering. Query both curriculum and challenges when a topic spans design theory and hands-on configuration, and use `read_full_page` if an excerpt is truncated.\n\n"
    "RESPONSE STYLE & COMPLEXITY:\n"
    "2. Answer directly without mandatory or formulaic headings (avoid boilerplate sections like 'Short answer', 'What the excerpts actually tell us', 'Implication for architecture and governance', or repetitive 'Bottom line' conclusions).\n"
    "3. Match depth to complexity: 80–150 words for simple definitions, 150–300 words for technical explanations/comparisons, and as detailed as necessary for complex architecture scenarios.\n"
    "4. Include practical Pega examples and architectural implications where useful.\n\n"
    "STRICT GROUNDING & CITATION INTEGRITY:\n"
    "5. Clearly distinguish retrieved curriculum facts (cited inline as `[filename, Page X, Section: Y]` using only files/pages returned by your tools) from general Pega knowledge. Never invent citations, attach unrelated sources, or present general knowledge as retrieved evidence.\n"
    "6. If retrieved excerpts are insufficient after searching, explicitly note that limitation and provide helpful general Pega guidance clearly labeled as general knowledge without citations."
)


def should_use_agent_loop(query: str) -> Tuple[bool, str]:
    """
    Lightweight query router:
    - Straightforward conceptual or comparison questions route to fast Hybrid + Reranked Direct Retrieval.
    - Multi-part, cross-pillar, or hands-on + theory investigative queries route to the multi-step Agent Loop.
    """
    q_low = query.lower().strip()
    investigative_signals = [
        "challenge", "walkthrough", "step-by-step", "step by step",
        "end-to-end", "end to end", "and also", "along with",
        "compare and implement", "design and configure", "full page", "page "
    ]
    multi_clause = (len(q_low.split()) >= 20 and (" and " in q_low or ";" in q_low or "?" in q_low[:-1]))
    if any(sig in q_low for sig in investigative_signals) or multi_clause:
        return True, "Multi-step investigative query detected; using autonomous tool-calling agent."
    return False, "Straightforward architectural query; using Hybrid Retrieval + Cross-Encoder Reranking."


def run_agent(
    engine: Any,
    query: str,
    groq_api_key: str,
    model_name: str = PRIMARY_GROQ_MODEL,
    temperature: float = 0.0,
    max_iterations: int = 4,
    chat_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = DEFAULT_FINAL_EVIDENCE_K,
    use_reranker: bool = True,
    dense_weight: float = 0.5,
    section_filter: Optional[str] = None,
    groq_factory: Callable[..., Any] = Groq,
    should_use_agent_fn: Callable[[str], Tuple[bool, str]] = should_use_agent_loop,
) -> Dict[str, Any]:
    """
    Runs an autonomous agent loop powered by Groq Function Calling:
    1. Agent plans which tools to call (curriculum search, challenge search, or full-page read).
    2. Executes tools locally against ChromaDB & PDFs.
    3. Feeds verified evidence back to the model.
    4. If a model does not support tool calling (or encounters a Groq 400), automatically
       falls back to Direct Grounded Retrieval without failing.
    """
    if not groq_api_key or not groq_api_key.strip():
        return {
            "answer": "⚠️ **Groq API Key Required:** Please provide your Groq API key in the sidebar to activate the Agent. You can get one free at [console.groq.com/keys](https://console.groq.com/keys).",
            "trace": [],
            "sources": []
        }

    client = groq_factory(api_key=groq_api_key.strip())
    model_name, model_notice = resolve_available_model(client, model_name)
    execution_trace: List[Dict[str, Any]] = []
    if model_notice:
        execution_trace.append({"type": "note", "message": model_notice})
    collected_sources: List[Dict[str, Any]] = []

    if not is_tool_capable_model(model_name):
        return execute_direct_grounded_fallback(
            engine=engine,
            query=query,
            client=client,
            model_name=model_name,
            temperature=temperature,
            chat_history=chat_history,
            execution_trace=execution_trace,
            reason=f"Model '{model_name}' does not support function/tool calling. Directly routed to Grounded Retrieval.",
            top_k=top_k,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
            section_filter=section_filter
        )

    use_agent, route_reason = should_use_agent_fn(query)
    if not use_agent:
        return execute_direct_grounded_fallback(
            engine=engine,
            query=query,
            client=client,
            model_name=model_name,
            temperature=temperature,
            chat_history=chat_history,
            execution_trace=execution_trace,
            reason=route_reason,
            top_k=top_k,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
            section_filter=section_filter
        )

    tools_schema = get_agent_tools_schema()
    spec = get_model_spec(model_name)
    loop_max_tokens = min(2000, spec.get("max_completion_tokens", 1500) + 500)

    messages: List[Any] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    if chat_history:
        for h in chat_history[-6:]:
            role = h.get("role", "user")
            if role in ("user", "assistant") and h.get("content"):
                messages.append({"role": role, "content": h["content"][:700]})
    messages.append({"role": "user", "content": query})

    current_iteration = 0
    while current_iteration < max_iterations:
        current_iteration += 1
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=loop_max_tokens
            )
        except Exception as e:
            err_str = str(e).lower()
            is_tool_err = any(x in err_str for x in [
                "tool calling is not supported",
                "tool calling",
                "tools is not supported",
                "does not support tools",
                "param': 'tool calling'",
                "param': 'tools'"
            ])
            if is_tool_err:
                return execute_direct_grounded_fallback(
                    engine=engine,
                    query=query,
                    client=client,
                    model_name=model_name,
                    temperature=temperature,
                    chat_history=chat_history,
                    execution_trace=execution_trace,
                    reason=f"Groq API reported tool calling unsupported for model '{model_name}'. Auto-switched to Direct Grounded Retrieval.",
                    top_k=top_k,
                    use_reranker=use_reranker,
                    dense_weight=dense_weight,
                    section_filter=section_filter
                )

            is_model_issue = any(x in err_str for x in [
                "model_not_found", "model_terms_required", "model_decommissioned",
                "decommissioned", "deprecated", "does not exist", "not have access"
            ])
            if is_model_issue:
                alt_model = find_best_alternative_model(client, model_name)
                if alt_model and alt_model != model_name:
                    model_name = alt_model
                    execution_trace.append({
                        "step": current_iteration,
                        "tool": "model_auto_switch",
                        "args": {"new_model": alt_model},
                        "summary": f"Requested model unavailable; auto-switched to `{alt_model}`."
                    })
                    if not is_tool_capable_model(alt_model):
                        return execute_direct_grounded_fallback(
                            engine=engine,
                            query=query,
                            client=client,
                            model_name=alt_model,
                            temperature=temperature,
                            chat_history=chat_history,
                            execution_trace=execution_trace,
                            reason=f"Auto-switched to `{alt_model}` using Direct Grounded Retrieval.",
                            top_k=top_k,
                            use_reranker=use_reranker,
                            dense_weight=dense_weight,
                            section_filter=section_filter
                        )
                    continue

            if is_context_length_error(err_str):
                return execute_direct_grounded_fallback(
                    engine=engine,
                    query=query,
                    client=client,
                    model_name=model_name,
                    temperature=temperature,
                    chat_history=chat_history,
                    execution_trace=execution_trace,
                    reason=f"Agent history exceeded context window for '{model_name}'; switched to Direct Grounded Retrieval.",
                    top_k=top_k,
                    use_reranker=use_reranker,
                    dense_weight=dense_weight,
                    section_filter=section_filter
                )

            if "rate_limit_exceeded" in err_str or "413" in err_str or "too large" in err_str:
                return execute_direct_grounded_fallback(
                    engine=engine,
                    query=query,
                    client=client,
                    model_name=model_name,
                    temperature=temperature,
                    chat_history=chat_history,
                    execution_trace=execution_trace,
                    reason="Payload size exceeded Groq rate limit; switched to compact Direct Grounded Retrieval.",
                    top_k=top_k,
                    use_reranker=use_reranker,
                    dense_weight=dense_weight,
                    section_filter=section_filter
                )

            return {
                "answer": f"❌ **Groq API Error during Agent Execution:** {str(e)}",
                "trace": execution_trace,
                "sources": collected_sources
            }

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            final_sources = deduplicate_sources(collected_sources)
            raw_ans = response_message.content or "No response generated."
            if getattr(response.choices[0], "finish_reason", None) == "length":
                raw_ans = trim_to_last_complete_sentence(raw_ans) + "\n\n> *Note: Response reached the model's output length limit and was trimmed at the last complete sentence.*"
            verified_ans, _ = verify_citations(raw_ans, final_sources)
            return {
                "answer": verified_ans,
                "trace": execution_trace,
                "sources": final_sources
            }

        messages.append(response_message)
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            func_args_str = tool_call.function.arguments
            try:
                parsed_args = json.loads(func_args_str or "{}")
                func_args = parsed_args if isinstance(parsed_args, dict) else {}
            except Exception:
                func_args = {}

            tool_output_dict: Dict[str, Any] = {}
            if func_name == "search_curriculum":
                tool_output_dict = engine.tool_search_curriculum(
                    query=func_args.get("query") or query,
                    section=func_args.get("section")
                )
                step_summary = f"Searched curriculum modules for \"{func_args.get('query', query)}\""
            elif func_name == "search_challenges":
                tool_output_dict = engine.tool_search_challenges(
                    query=func_args.get("query") or query
                )
                step_summary = f"Searched hands-on challenges for \"{func_args.get('query', query)}\""
            elif func_name == "read_full_page":
                try:
                    page_num_arg = int(func_args.get("page_number") or 1)
                except (TypeError, ValueError):
                    page_num_arg = 1
                tool_output_dict = engine.tool_read_full_page(
                    document_name=func_args.get("document_name") or "",
                    page_number=page_num_arg
                )
                step_summary = f"Read full page {func_args.get('page_number', 1)} of {func_args.get('document_name', 'document')}"
            elif func_name == "lookup_pega_terms":
                tool_output_dict = engine.tool_lookup_pega_terms(
                    term=func_args.get("term") or query
                )
                step_summary = f"Looked up Pega rule/term \"{func_args.get('term', query)}\""
            else:
                tool_output_dict = {"status": "error", "message": f"Unknown tool '{func_name}'"}
                step_summary = f"Executed `{func_name}`"

            if not isinstance(tool_output_dict, dict):
                tool_output_dict = {}

            collect_tool_sources(tool_output_dict, collected_sources, include_preview=True)

            execution_trace.append({
                "step": current_iteration,
                "tool": func_name,
                "args": func_args,
                "summary": step_summary
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": json.dumps(tool_output_dict, ensure_ascii=False)
            })

    try:
        if collected_sources:
            unique_sources = deduplicate_sources(collected_sources)
            final_answer = generate_answer(
                query=query,
                retrieved_chunks=unique_sources,
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=temperature,
                chat_history=chat_history,
                groq_factory=groq_factory
            )
            if isinstance(final_answer, str) and final_answer.startswith("❌"):
                return execute_direct_grounded_fallback(
                    engine=engine,
                    query=query,
                    client=client,
                    model_name=model_name,
                    temperature=temperature,
                    chat_history=chat_history,
                    execution_trace=execution_trace,
                    reason=f"Synthesis fallback triggered: {final_answer}",
                    top_k=top_k,
                    use_reranker=use_reranker,
                    dense_weight=dense_weight,
                    section_filter=section_filter
                )
            return {
                "answer": final_answer,
                "trace": execution_trace,
                "sources": unique_sources
            }
        return execute_direct_grounded_fallback(
            engine=engine,
            query=query,
            client=client,
            model_name=model_name,
            temperature=temperature,
            chat_history=chat_history,
            execution_trace=execution_trace,
            reason="Research iterations completed without retrieving evidence.",
            top_k=top_k,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
            section_filter=section_filter
        )
    except Exception as e:
        try:
            return execute_direct_grounded_fallback(
                engine=engine,
                query=query,
                client=client,
                model_name=model_name,
                temperature=temperature,
                chat_history=chat_history,
                execution_trace=execution_trace,
                reason=f"Final pass recovery: {type(e).__name__}",
                top_k=top_k,
                use_reranker=use_reranker,
                dense_weight=dense_weight,
                section_filter=section_filter
            )
        except Exception:
            return {
                "answer": "❌ **Generation Error:** Unable to synthesize response from retrieved sources. Please verify your Groq connection and try again.",
                "trace": execution_trace,
                "sources": collected_sources
            }


def run_agent_stream(
    engine: Any,
    query: str,
    groq_api_key: str,
    model_name: str = PRIMARY_GROQ_MODEL,
    temperature: float = 0.0,
    max_iterations: int = 4,
    chat_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = DEFAULT_FINAL_EVIDENCE_K,
    use_reranker: bool = True,
    dense_weight: float = 0.5,
    section_filter: Optional[str] = None,
    groq_factory: Callable[..., Any] = Groq,
    should_use_agent_fn: Callable[[str], Tuple[bool, str]] = should_use_agent_loop,
):
    """
    Streaming-enabled Agentic & Hybrid RAG pipeline.
    Yields:
      - {"type": "status", "message": str}
      - {"type": "token", "text": str}
      - {"type": "final", "answer": str, "sources": list, "trace": list}
    """
    if not groq_api_key or not groq_api_key.strip():
        yield {
            "type": "final",
            "answer": "⚠️ **Groq API Key Required:** Please provide your Groq API key in Configuration to activate the Assistant.",
            "sources": [],
            "trace": [],
            "model_used": model_name or PRIMARY_GROQ_MODEL
        }
        return

    client = groq_factory(api_key=groq_api_key.strip())
    model_name, model_notice = resolve_available_model(client, model_name)
    execution_trace: List[Dict[str, Any]] = []
    if model_notice:
        execution_trace.append({"type": "note", "message": model_notice})
    collected_sources: List[Dict[str, Any]] = []

    use_agent, _ = should_use_agent_fn(query)
    if (not is_tool_capable_model(model_name)) or (not use_agent):
        yield {"type": "status", "message": "Retrieving Pega curriculum excerpts (Hybrid BM25 + Vector + Reranker)..."}
        search_query = contextualize_query_with_history(query, chat_history)
        retrieved_chunks = engine.search(
            query=search_query,
            top_k=top_k,
            section_filter=section_filter,
            use_reranker=use_reranker,
            dense_weight=dense_weight
        )
        if any(w in search_query.lower() for w in ["how to", "configure", "implement", "step", "challenge", "exercise", "create", "setup"]):
            try:
                ch_res = engine.tool_search_challenges(query=search_query, top_k=2)
                for r in ch_res.get("results", []):
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

        if not retrieved_chunks:
            yield {
                "type": "final",
                "answer": "The provided Pega curriculum documentation does not contain relevant excerpts for this query.",
                "sources": [],
                "trace": execution_trace
            }
            return

        yield {"type": "status", "message": "Generating grounded response..."}
        for ev in stream_answer(
            query=query,
            retrieved_chunks=retrieved_chunks,
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            chat_history=chat_history,
            groq_factory=groq_factory
        ):
            if ev["type"] == "token":
                yield ev
            elif ev["type"] == "final":
                yield {
                    "type": "final",
                    "answer": ev["answer"],
                    "sources": ev["sources"],
                    "trace": execution_trace
                }
        return

    tools_schema = get_agent_tools_schema()
    spec = get_model_spec(model_name)
    loop_max_tokens = min(2000, spec.get("max_completion_tokens", 1500) + 500)

    messages: List[Any] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    if chat_history:
        for h in chat_history[-6:]:
            role = h.get("role", "user")
            if role in ("user", "assistant") and h.get("content"):
                messages.append({"role": role, "content": h["content"][:700]})
    messages.append({"role": "user", "content": query})

    current_iteration = 0
    while current_iteration < max_iterations:
        current_iteration += 1
        yield {"type": "status", "message": f"Researching Pega curriculum (step {current_iteration}/{max_iterations})..."}

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=loop_max_tokens
            )
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ["tool calling", "tools is not supported", "does not support tools", "param': 'tools'"]):
                yield {"type": "status", "message": "Model does not support tool calling; switching to Direct Grounded Streaming..."}
                chunks = engine.search(
                    query=contextualize_query_with_history(query, chat_history),
                    top_k=top_k,
                    section_filter=section_filter,
                    use_reranker=use_reranker,
                    dense_weight=dense_weight
                )
                for ev in stream_answer(
                    query=query,
                    retrieved_chunks=chunks,
                    groq_api_key=groq_api_key,
                    model_name=model_name,
                    temperature=temperature,
                    chat_history=chat_history,
                    groq_factory=groq_factory
                ):
                    if ev["type"] == "token":
                        yield ev
                    elif ev["type"] == "final":
                        yield {"type": "final", "answer": ev["answer"], "sources": ev["sources"], "trace": execution_trace}
                return
            yield {"type": "final", "answer": f"❌ **Groq API Error:** {str(e)}", "sources": deduplicate_sources(collected_sources), "trace": execution_trace}
            return

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            final_sources = deduplicate_sources(collected_sources)
            raw_ans = response_message.content or "No response generated."
            if getattr(response.choices[0], "finish_reason", None) == "length":
                raw_ans = trim_to_last_complete_sentence(raw_ans) + "\n\n> *Note: Response reached the model's output length limit and was trimmed at the last complete sentence.*"
            verified_ans, _ = verify_citations(raw_ans, final_sources)
            yield {"type": "final", "answer": verified_ans, "sources": final_sources, "trace": execution_trace}
            return

        messages.append(response_message)
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            try:
                parsed_args = json.loads(tool_call.function.arguments or "{}")
                func_args = parsed_args if isinstance(parsed_args, dict) else {}
            except Exception:
                func_args = {}

            if func_name == "search_curriculum":
                q_arg = func_args.get("query") or query
                yield {"type": "status", "message": f"Searching curriculum for \"{q_arg}\"..."}
                tool_output_dict = engine.tool_search_curriculum(query=q_arg, section=func_args.get("section"))
            elif func_name == "search_challenges":
                q_arg = func_args.get("query") or query
                yield {"type": "status", "message": f"Searching hands-on challenges for \"{q_arg}\"..."}
                tool_output_dict = engine.tool_search_challenges(query=q_arg)
            elif func_name == "read_full_page":
                try:
                    page_num_arg = int(func_args.get("page_number") or 1)
                except (TypeError, ValueError):
                    page_num_arg = 1
                doc_arg = func_args.get("document_name") or ""
                yield {"type": "status", "message": f"Reading page {page_num_arg} of {doc_arg or 'PDF'}..."}
                tool_output_dict = engine.tool_read_full_page(document_name=doc_arg, page_number=page_num_arg)
            elif func_name == "lookup_pega_terms":
                term_arg = func_args.get("term") or query
                yield {"type": "status", "message": f"Looking up Pega term \"{term_arg}\"..."}
                tool_output_dict = engine.tool_lookup_pega_terms(term=term_arg)
            else:
                tool_output_dict = {"status": "error", "message": f"Unknown tool '{func_name}'"}

            if not isinstance(tool_output_dict, dict):
                tool_output_dict = {}

            collect_tool_sources(tool_output_dict, collected_sources, include_preview=False)

            execution_trace.append({
                "step": current_iteration,
                "tool": func_name,
                "args": func_args,
                "summary": f"Executed {func_name}"
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": json.dumps(tool_output_dict, ensure_ascii=False)
            })

    unique_sources = deduplicate_sources(collected_sources)
    if not unique_sources:
        unique_sources = engine.search(
            query=contextualize_query_with_history(query, chat_history),
            top_k=top_k,
            section_filter=section_filter,
            use_reranker=use_reranker,
            dense_weight=dense_weight
        )

    yield {"type": "status", "message": "Synthesizing verified architectural response..."}
    for ev in stream_answer(
        query=query,
        retrieved_chunks=unique_sources,
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
        chat_history=chat_history,
        groq_factory=groq_factory
    ):
        if ev["type"] == "token":
            yield ev
        elif ev["type"] == "final":
            yield {"type": "final", "answer": ev["answer"], "sources": ev["sources"], "trace": execution_trace}
