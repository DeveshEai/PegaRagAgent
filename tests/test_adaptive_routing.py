from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import rag_engine


def test_stage1_router_config_defaults():
    assert rag_engine.ROUTER_MODEL == "qwen/qwen3.8-27b"
    assert rag_engine.ENABLE_ADAPTIVE_ROUTING is False
    assert rag_engine.MAX_ROUTING_ESCALATIONS == 1


def test_deterministic_fast_paths_and_non_tool_model_protection(isolated_env):
    engine = isolated_env["engine"]

    # Non-tool-capable answer models must always stay on DIRECT_RAG without LLM router calls
    for direct_model in ("allam-2-7b", "qwen/qwen3.8-27b"):
        res = engine.classify_query_intent(
            "Walkthrough challenge step-by-step for Queue Processor",
            groq_api_key="gsk_test",
            answer_model_name=direct_model,
        )
        assert res["route"] == "DIRECT_RAG"
        assert res["tier"] == "deterministic"
        assert res["tokens_used"] == 0

    # Short standalone definition -> SIMPLE_LOOKUP / DIRECT_RAG
    res_simple = engine.classify_query_intent(
        "What is pyWorkPage?",
        groq_api_key="gsk_test",
        answer_model_name="openai/gpt-oss-120b",
    )
    assert res_simple["intent"] == "SIMPLE_LOOKUP"
    assert res_simple["route"] == "DIRECT_RAG"
    assert res_simple["tier"] == "deterministic"

    # Explicit challenge / page lookup -> STEP_BY_STEP_CHALLENGE / AUTONOMOUS_AGENT
    res_chal = engine.classify_query_intent(
        "Walkthrough challenge for configuring SAML",
        groq_api_key="gsk_test",
        answer_model_name="openai/gpt-oss-120b",
    )
    assert res_chal["intent"] == "STEP_BY_STEP_CHALLENGE"
    assert res_chal["route"] == "AUTONOMOUS_AGENT"
    assert res_chal["tier"] == "deterministic"


def test_llm_router_comparison_flexibility_followup_context_and_fallback(isolated_env):
    engine = isolated_env["engine"]
    captured_payloads = []

    def _make_resp(content_str: str, tokens: int = 85):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content_str))],
            usage=SimpleNamespace(total_tokens=tokens),
        )

    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(id="qwen/qwen3.8-27b"),
                SimpleNamespace(id="openai/gpt-oss-120b"),
                SimpleNamespace(id="openai/gpt-oss-20b"),
            ]
        )

        def _fake_create(**kwargs):
            captured_payloads.append(kwargs)
            user_msg = kwargs["messages"][-1]["content"]
            if "enterprise" in user_msg.lower():
                return _make_resp('{"intent":"COMPARISON","route":"AUTONOMOUS_AGENT","reason":"Multi-document architectural trade-off"}')
            return _make_resp('{"intent":"COMPARISON","route":"DIRECT_RAG","reason":"Single-topic comparison"}')

        mock_client.chat.completions.create.side_effect = _fake_create
        mock_groq_cls.return_value = mock_client

        # Straightforward comparison routes deterministically to DIRECT_RAG without an LLM call
        res_simple_cmp = engine.classify_query_intent(
            "Compare Queue Processor vs Job Scheduler",
            groq_api_key="gsk_test",
        )
        assert res_simple_cmp["intent"] == "COMPARISON"
        assert res_simple_cmp["route"] == "DIRECT_RAG"
        assert res_simple_cmp["tier"] == "deterministic"
        assert res_simple_cmp["tokens_used"] == 0

        # Ambiguous queries use the qwen/qwen3.8-27b LLM router
        res_ambig = engine.classify_query_intent(
            "Should we use a Queue Processor or Job Scheduler for our nightly batch?",
            groq_api_key="gsk_test",
        )
        assert res_ambig["intent"] == "COMPARISON"
        assert res_ambig["route"] == "DIRECT_RAG"
        assert res_ambig["tier"] == "llm"
        assert res_ambig["model_used"] == "qwen/qwen3.8-27b"
        assert res_ambig["tokens_used"] == 85

        res_complex_cmp = engine.classify_query_intent(
            "Compare enterprise class structure and deployment strategies across multiple applications",
            groq_api_key="gsk_test",
        )
        assert res_complex_cmp["route"] == "AUTONOMOUS_AGENT"

        # Short follow-up preserves conversation history in router prompt
        history = [
            {"role": "user", "content": "Explain Queue Processors in Pega."},
            {"role": "assistant", "content": "Queue Processors handle async background work."},
        ]
        res_followup = engine.classify_query_intent(
            "What about retries?",
            chat_history=history,
            groq_api_key="gsk_test",
        )
        assert res_followup["tier"] == "llm"
        assert "Queue Processors" in captured_payloads[-1]["messages"][-1]["content"]

    # Router fallback when qwen/qwen3.8-27b is unavailable -> openai/gpt-oss-20b,
    # and safe deterministic fallback on malformed JSON
    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="openai/gpt-oss-20b")]
        )
        mock_client.chat.completions.create.side_effect = [
            _make_resp('{"intent":"TECHNICAL_EXPLANATION","route":"DIRECT_RAG","reason":"ok"}', tokens=120),
            _make_resp("NOT VALID JSON AT ALL"),
        ]
        mock_groq_cls.return_value = mock_client

        res_fb_model = engine.classify_query_intent(
            "When should ABAC replace RBAC in a multi-tenant Pega app?",
            groq_api_key="gsk_test",
        )
        assert res_fb_model["model_used"] == "openai/gpt-oss-20b"
        assert res_fb_model["route"] == "DIRECT_RAG"

        res_bad_json = engine.classify_query_intent(
            "Should we configure a savable Data Page for this workflow?",
            groq_api_key="gsk_test",
        )
        assert res_bad_json["tier"] == "fallback"
        assert res_bad_json["route"] == "DIRECT_RAG"


def test_pega_domain_taxonomy_and_complexity_signals():
    from agents.research_agent import PEGA_COMPLEXITY_SIGNALS, PEGA_DOMAIN_TERMS

    expected_domains = {
        "data_pages", "clipboard", "rule_resolution", "circumstancing",
        "case_management", "sla", "queue_processor", "job_scheduler",
        "integration", "security", "constellation", "data_management",
    }
    expected_complexity = {
        "architecture_design", "troubleshooting", "root_cause_analysis",
        "dependency_impact", "multi_rule_investigation", "implementation_walkthrough",
    }
    assert expected_domains.issubset(set(PEGA_DOMAIN_TERMS.keys()))
    assert expected_complexity.issubset(set(PEGA_COMPLEXITY_SIGNALS.keys()))


def test_pega_domain_deterministic_routing_matrix(monkeypatch):
    """Verify 20+ representative Pega queries including false-positive and false-negative cases."""
    import rag_engine

    monkeypatch.setattr(rag_engine, "PegaRAGEngine", rag_engine.PegaRAGEngine)
    engine = rag_engine.PegaRAGEngine.__new__(rag_engine.PegaRAGEngine)

    # 1. False-positive guards: Pega terms and 'and' must NOT blindly trigger AUTONOMOUS_AGENT
    direct_rag_queries = [
        "What is Lock and Roll in Pega?",
        "What is a Data Page in Pega?",
        "Define PyWorkPage on the Clipboard",
        "Explain Rule Resolution in Pega",
        "How does Circumstancing work in Pega?",
        "What is an SLA goal and deadline in Pega?",
        "Explain Queue Processor and Job Scheduler in Pega",
        "What is the difference between RBAC and ABAC in Pega?",
        "Compare Data Page vs Clipboard page",
        "Compare Queue Processor vs Job Scheduler",
        "What is Constellation in Pega?",
        "Explain Connect-REST and auth profile in Pega",
        "What is a Data Transform in Pega?",
        "Explain Case Management stages and processes",
    ]
    for q in direct_rag_queries:
        res = engine.classify_query_intent(q, groq_api_key="gsk_unused")
        assert res["route"] == "DIRECT_RAG", f"Expected DIRECT_RAG for: {q}, got {res}"
        assert res["tier"] == "deterministic", f"Expected deterministic fast-path for: {q}, got {res}"

    # 2. False-negative guards: genuine complexity/investigative tasks MUST route to AUTONOMOUS_AGENT
    agent_queries = [
        "Troubleshoot why my Queue Processor is stuck and failing with Data Page errors",
        "What is the root cause of stale Clipboard data after an SLA escalation?",
        "Analyze the downstream impact of changing class inheritance in Rule Resolution",
        "Walkthrough step-by-step how to build a Constellation case type with Connect-REST",
        "Explain how Case Management, SLA, and Queue Processor interact across multiple rules",
        "Architect a multi-application enterprise design for security ABAC and DX API integration",
        "Complete the hands-on challenge for Circumstancing and Job Scheduler",
        "Diagnose why our savable Data Page throws a case locking exception on commit",
    ]
    for q in agent_queries:
        res = engine.classify_query_intent(q, groq_api_key="gsk_unused")
        assert res["route"] == "AUTONOMOUS_AGENT", f"Expected AUTONOMOUS_AGENT for: {q}, got {res}"
        assert res["tier"] == "deterministic", f"Expected deterministic fast-path for: {q}, got {res}"

    # 3. Verify should_use_agent_loop directly does not trigger on long 'and' explanations
    use_agent_fp, _ = rag_engine.should_use_agent_loop(
        "Explain how Data Pages and Clipboard pages store customer data and refresh strategies during a normal case lifecycle"
    )
    assert use_agent_fp is False

    use_agent_fn, _ = rag_engine.should_use_agent_loop(
        "What is the root cause when a Data Page and Queue Processor fail during case locking?"
    )
    assert use_agent_fn is True


def test_stage2_adaptive_non_streaming_paths_and_error_handling(isolated_env):
    """Verify Stage 2 run_adaptive connects DIRECT_RAG, AUTONOMOUS_AGENT, manual modes, and error fallbacks."""
    from unittest.mock import MagicMock, patch
    from types import SimpleNamespace
    import rag_engine

    engine = isolated_env["engine"]
    sample_chunks = [
        {
            "source": "CLSA_Guide.pdf",
            "page": "12",
            "section": "Data Pages",
            "category": "Curriculum",
            "score": 91.5,
            "document": "A Data Page caches data on demand on the Clipboard.",
        }
    ]
    engine.search = MagicMock(return_value=sample_chunks)
    engine.tool_search_curriculum = MagicMock(
        return_value={
            "results": [
                {
                    "document": "CLSA_Guide.pdf",
                    "page": 12,
                    "section": "Queue Processor",
                    "relevance_score": "92.0%",
                    "excerpt": "Queue Processors process items asynchronously.",
                }
            ]
        }
    )

    def _msg_resp(content, tool_calls=None):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=content, tool_calls=tool_calls),
                )
            ],
            usage=SimpleNamespace(total_tokens=42),
        )

    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(id="openai/gpt-oss-120b"),
                SimpleNamespace(id="qwen/qwen3.8-27b"),
            ]
        )
        # 1) Direct RAG path call (grounded answer)
        # 2) Agent path call: 1 tool call + 1 final answer
        tool_call = SimpleNamespace(
            id="tc_1",
            function=SimpleNamespace(
                name="search_curriculum",
                arguments='{"query":"Queue Processor broken queue","top_k":3}',
            ),
        )
        mock_client.chat.completions.create.side_effect = [
            _msg_resp("A Data Page caches data on demand [CLSA_Guide.pdf, Page 12, Section: Data Pages]."),
            _msg_resp("", tool_calls=[tool_call]),
            _msg_resp("Root cause involves broken queue retries [CLSA_Guide.pdf, Page 12, Section: Queue Processor]."),
        ]
        mock_groq_cls.return_value = mock_client

        # DIRECT_RAG path (deterministic, no router LLM call)
        res_direct = engine.run_adaptive(
            "What is a Data Page in Pega?",
            groq_api_key="gsk_test",
        )
        assert res_direct["routing"]["route"] == "DIRECT_RAG"
        assert res_direct["routing"]["tier"] == "deterministic"
        assert "Data Page" in res_direct["answer"]
        assert len(res_direct["sources"]) >= 1
        assert res_direct["trace"][0]["tool"] == "direct_grounded_retrieval"
        engine.search.assert_called_once()

        # AUTONOMOUS_AGENT path
        res_agent = engine.run_adaptive(
            "Troubleshoot why my Queue Processor is stuck and failing with Data Page errors",
            groq_api_key="gsk_test",
        )
        assert res_agent["routing"]["route"] == "AUTONOMOUS_AGENT"
        assert res_agent["trace"][0]["tool"] == "search_curriculum"
        assert "broken queue" in res_agent["answer"].lower()
        engine.tool_search_curriculum.assert_called_once()

    # Missing API key error handling
    res_no_key = engine.run_adaptive("What is a Data Page in Pega?", groq_api_key="")
    assert "Groq API Key Required" in res_no_key["answer"]
    assert res_no_key["routing"]["tier"] == "fallback"

    # Manual mode override + router exception fallback
    from agents.research_agent import run_adaptive as _run_adaptive_fn

    with patch("agents.research_agent.run_agent", return_value={"answer": "ok", "trace": [], "sources": []}) as mock_ra:
        res_manual = _run_adaptive_fn(
            engine=engine,
            query="Troubleshoot why Queue Processor fails",
            groq_api_key="gsk_test",
            execution_mode="direct",
        )
        assert res_manual["routing"]["route"] == "DIRECT_RAG"
        assert res_manual["routing"]["tier"] == "manual"
        mock_ra.assert_called_once()

        def _boom(**_kw):
            raise RuntimeError("router timeout")

        res_err = _run_adaptive_fn(
            engine=engine,
            query="What is a Data Page in Pega?",
            groq_api_key="gsk_test",
            execution_mode="adaptive",
            classify_fn=_boom,
        )
        assert res_err["routing"]["tier"] == "fallback"
        assert res_err["routing"]["route"] == "DIRECT_RAG"


def test_stage2_adaptive_streaming_paths_and_completeness(isolated_env):
    """Verify Stage 2 run_adaptive_stream preserves streaming tokens, citations, history, and routing metadata."""
    from unittest.mock import MagicMock, patch
    from types import SimpleNamespace
    import rag_engine

    engine = isolated_env["engine"]
    sample_chunks = [
        {
            "source": "CLSA_Guide.pdf",
            "page": "15",
            "section": "Circumstancing",
            "category": "Curriculum",
            "score": 89.0,
            "document": "Circumstancing specializes rules based on property or date conditions.",
        }
    ]
    engine.search = MagicMock(return_value=sample_chunks)
    engine.tool_search_curriculum = MagicMock(
        return_value={
            "results": [
                {
                    "document": "CLSA_Guide.pdf",
                    "page": 20,
                    "section": "Rule Resolution",
                    "relevance_score": "90.0%",
                    "excerpt": "Rule resolution selects the best matching rule.",
                }
            ]
        }
    )

    def _stream_chunks(text):
        for tok in text.split(" "):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=tok + " "), finish_reason=None)])
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=""), finish_reason="stop")])

    with patch("rag_engine.Groq") as mock_groq_cls, patch("agents.answer_generator.Groq") as mock_ag_groq:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="openai/gpt-oss-120b")]
        )

        # Streaming DIRECT_RAG test
        mock_client.chat.completions.create.return_value = _stream_chunks(
            "Circumstancing specializes rules [CLSA_Guide.pdf, Page 15, Section: Circumstancing]."
        )
        mock_groq_cls.return_value = mock_client
        mock_ag_groq.return_value = mock_client

        events_direct = list(
            engine.run_adaptive_stream(
                "How does Circumstancing work in Pega?",
                groq_api_key="gsk_test",
            )
        )
        statuses = [e["message"] for e in events_direct if e.get("type") == "status"]
        tokens = [e["text"] for e in events_direct if e.get("type") == "token"]
        finals = [e for e in events_direct if e.get("type") == "final"]
        assert any("Adaptive route: DIRECT_RAG" in s for s in statuses)
        assert len(tokens) > 0
        assert len(finals) == 1
        assert finals[0]["routing"]["route"] == "DIRECT_RAG"
        assert len(finals[0]["sources"]) == 1

        # Streaming AUTONOMOUS_AGENT test
        tool_call = SimpleNamespace(
            id="tc_stream_1",
            function=SimpleNamespace(name="search_curriculum", arguments='{"query":"Rule Resolution impact","top_k":3}'),
        )
        agent_msg_resp = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="", tool_calls=[tool_call]))]
        )
        agent_done_resp = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="Ready to synthesize.", tool_calls=None))]
        )
        mock_client.chat.completions.create.side_effect = [
            agent_msg_resp,
            agent_done_resp,
            _stream_chunks("Impact analysis complete [CLSA_Guide.pdf, Page 20, Section: Rule Resolution]."),
        ]
        events_agent = list(
            engine.run_adaptive_stream(
                "Analyze the downstream impact of changing class inheritance in Rule Resolution",
                groq_api_key="gsk_test",
            )
        )
        finals_agent = [e for e in events_agent if e.get("type") == "final"]
        assert len(finals_agent) == 1
        assert finals_agent[0]["routing"]["route"] == "AUTONOMOUS_AGENT"
        assert finals_agent[0]["trace"][0]["tool"] == "search_curriculum"


def test_stage3_evidence_sufficiency_evaluator():
    """Verify evaluate_evidence_sufficiency does not rely on source count or score alone."""
    import rag_engine

    # High score & multiple chunks, but irrelevant to query terms -> insufficient
    irrelevant_chunks = [
        {"source": "DocA.pdf", "page": "1", "score": 95.0, "document": "Unrelated UI theme styling guidelines for portals."},
        {"source": "DocB.pdf", "page": "2", "score": 92.0, "document": "Another unrelated paragraph about CSS colors and fonts."},
    ]
    eval_irrel = rag_engine.evaluate_evidence_sufficiency(
        "What is a Queue Processor in Pega?", irrelevant_chunks, intent="SIMPLE_LOOKUP"
    )
    assert eval_irrel["sufficient"] is False

    # Sufficient quality, score, diversity, and query coverage
    good_chunks = [
        {
            "source": "CLSA_Guide.pdf",
            "page": "10",
            "score": 88.0,
            "document": "A Queue Processor handles asynchronous background processing in Pega using Kafka.",
        }
    ]
    eval_good = rag_engine.evaluate_evidence_sufficiency(
        "What is a Queue Processor in Pega?", good_chunks, intent="SIMPLE_LOOKUP"
    )
    assert eval_good["sufficient"] is True


def test_stage3_escalation_one_limit_non_tool_guard_and_knowledge_gap(isolated_env):
    """Verify Stage 3 escalation, 1-escalation cap, non-tool model guard, and honest knowledge-gap answers."""
    from unittest.mock import MagicMock, patch
    from types import SimpleNamespace
    import rag_engine

    engine = isolated_env["engine"]

    def _msg_resp(content, tool_calls=None):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=content, tool_calls=tool_calls),
                )
            ],
            usage=SimpleNamespace(total_tokens=30),
        )

    # 1. Insufficient Direct RAG -> escalates ONCE to Autonomous Agent and succeeds
    engine.search = MagicMock(return_value=[])
    engine.tool_search_curriculum = MagicMock(
        return_value={
            "results": [
                {
                    "document": "CLSA_Guide.pdf",
                    "page": 44,
                    "section": "Data Pages",
                    "relevance_score": "91.0%",
                    "excerpt": "A Data Page loads and caches clipboard data on demand.",
                }
            ]
        }
    )

    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(id="openai/gpt-oss-120b"),
                SimpleNamespace(id="allam-2-7b"),
            ]
        )
        tool_call = SimpleNamespace(
            id="tc_esc_1",
            function=SimpleNamespace(name="search_curriculum", arguments='{"query":"Data Page","top_k":3}'),
        )
        mock_client.chat.completions.create.side_effect = [
            _msg_resp("", tool_calls=[tool_call]),
            _msg_resp("A Data Page caches clipboard data [CLSA_Guide.pdf, Page 44, Section: Data Pages]."),
        ]
        mock_groq_cls.return_value = mock_client

        res_esc = engine.run_adaptive("What is a Data Page in Pega?", groq_api_key="gsk_test")
        assert res_esc["routing"]["escalated"] is True
        assert res_esc["routing"]["escalation_count"] == 1
        assert res_esc["routing"]["final_route"] == "AUTONOMOUS_AGENT"
        assert "Data Page" in res_esc["answer"]
        assert len(res_esc["sources"]) == 1

    # 2. Failed escalation (Agent also finds 0 sources) -> max 1 escalation, honest knowledge-gap answer
    engine.search = MagicMock(return_value=[])
    engine.tool_search_curriculum = MagicMock(return_value={"results": []})
    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="openai/gpt-oss-120b")]
        )
        tool_call = SimpleNamespace(
            id="tc_esc_2",
            function=SimpleNamespace(name="search_curriculum", arguments='{"query":"ObscureRule","top_k":3}'),
        )
        mock_client.chat.completions.create.side_effect = [
            _msg_resp("", tool_calls=[tool_call]),
            _msg_resp("Here is some guessed info."),
        ]
        mock_groq_cls.return_value = mock_client

        res_gap = engine.run_adaptive("What is ObscureRule in Pega?", groq_api_key="gsk_test")
        assert res_gap["routing"]["escalated"] is True
        assert res_gap["routing"]["escalation_count"] <= rag_engine.MAX_ROUTING_ESCALATIONS
        assert "Insufficient Verified Evidence" in res_gap["answer"]

    # 3. Non-tool-capable model -> NEVER escalates, returns honest knowledge-gap answer
    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="allam-2-7b")]
        )
        mock_groq_cls.return_value = mock_client

        res_nontool = engine.run_adaptive(
            "What is ObscureRule in Pega?",
            groq_api_key="gsk_test",
            model_name="allam-2-7b",
        )
        assert res_nontool["routing"]["escalated"] is False
        assert res_nontool["routing"]["escalation_count"] == 0
        assert "Insufficient Verified Evidence" in res_nontool["answer"]
        assert "does not support autonomous tool-calling escalation" in res_nontool["answer"]


def test_stage3_streaming_escalation_and_knowledge_gap(isolated_env):
    """Verify Stage 3 run_adaptive_stream emits escalation status and honest knowledge-gap fallback."""
    from unittest.mock import MagicMock, patch
    from types import SimpleNamespace
    import rag_engine

    engine = isolated_env["engine"]
    engine.search = MagicMock(return_value=[])
    engine.tool_search_curriculum = MagicMock(return_value={"results": []})

    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="openai/gpt-oss-120b")]
        )
        tool_call = SimpleNamespace(
            id="tc_stream_gap",
            function=SimpleNamespace(name="search_curriculum", arguments='{"query":"UnknownRule","top_k":3}'),
        )
        mock_client.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="", tool_calls=[tool_call]))]),
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="Done", tool_calls=None))]),
            iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=""), finish_reason="stop")])]),
        ]
        mock_groq_cls.return_value = mock_client

        events = list(engine.run_adaptive_stream("What is UnknownRule in Pega?", groq_api_key="gsk_test"))
        statuses = [e["message"] for e in events if e.get("type") == "status"]
        finals = [e for e in events if e.get("type") == "final"]
        assert any("escalating to Autonomous CLSA Agent" in s for s in statuses)
        assert len(finals) == 1
        assert finals[0]["routing"]["escalated"] is True
        assert finals[0]["routing"]["escalation_count"] == 1
        assert "Insufficient Verified Evidence" in finals[0]["answer"]


def test_stage4_routing_evaluation_suite_and_report():
    """Verify Stage 4 representative Pega evaluation dataset, metrics, and Markdown report."""
    import rag_engine

    expected_categories = {
        "definitions",
        "technical_rules",
        "comparisons",
        "architecture",
        "troubleshooting",
        "integrations",
        "data_pages",
        "case_management",
        "hands_on_challenges",
    }
    dataset_categories = {item["category"] for item in rag_engine.PEGA_ROUTING_EVAL_DATASET}
    assert expected_categories.issubset(dataset_categories)

    summary = rag_engine.evaluate_routing_suite()
    assert summary["total_cases"] >= 24
    assert summary["routing_accuracy"] == 100.0
    assert summary["unnecessary_agent_calls"] == 0
    assert summary["missed_escalations"] == 0

    report = rag_engine.format_routing_eval_report(summary)
    assert "# Adaptive RAG Routing Evaluation Report" in report
    assert "100.0%" in report
    assert "Category Breakdown" in report


def test_stage4_observability_redaction_and_tracking(isolated_env):
    """Verify Stage 4 observability tracks routing fields and redacts API keys/cookies."""
    from unittest.mock import MagicMock, patch
    from types import SimpleNamespace
    import rag_engine
    from agents.routing_eval import ROUTING_TRACKER

    ROUTING_TRACKER.clear()

    # Secret redaction check
    sanitized = rag_engine.sanitize_routing_record(
        "Explain Data Page with api_key=gsk_supersecret1234567890 cookie=session_abc123",
        {
            "intent": "SIMPLE_LOOKUP",
            "route": "DIRECT_RAG",
            "tier": "deterministic",
            "latency_ms": 1.25,
            "tokens_used": 0,
            "escalated": False,
            "escalation_count": 0,
            "evidence_evaluation": {"sufficient": True},
        },
    )
    assert "gsk_supersecret" not in sanitized["query_preview"]
    assert "session_abc123" not in sanitized["query_preview"]
    assert "[REDACTED]" in sanitized["query_preview"]

    # End-to-end telemetry emission from run_adaptive
    engine = isolated_env["engine"]
    engine.search = MagicMock(
        return_value=[
            {
                "source": "CLSA.pdf",
                "page": "5",
                "score": 90.0,
                "document": "A Data Page loads and caches data on demand on the Clipboard.",
            }
        ]
    )
    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="openai/gpt-oss-120b")]
        )
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="A Data Page caches data [CLSA.pdf, Page 5, Section: General]."),
                )
            ]
        )
        mock_groq_cls.return_value = mock_client

        engine.run_adaptive("What is a Data Page in Pega?", groq_api_key="gsk_live_secret_value_1234")

    obs = rag_engine.get_routing_observability_summary()
    assert obs["total_queries"] == 1
    assert obs["direct_rag_count"] == 1
    assert obs["escalation_count"] == 0
    assert "gsk_live_secret" not in str(ROUTING_TRACKER.records())


def test_ux_conversational_fast_path_bypasses_retrieval_and_preserves_short_pega(isolated_env):
    """Greetings bypass retrieval/citations in shared execution layer, while short Pega queries still retrieve."""
    from unittest.mock import MagicMock
    import rag_engine

    engine = isolated_env["engine"]
    engine.search = MagicMock(return_value=[])

    for greeting in ("Hi", "Hey!", "Hello", "Thanks", "Who are you?"):
        cls_res = engine.classify_query_intent(greeting)
        assert cls_res["intent"] == "CONVERSATIONAL"
        assert cls_res["route"] == "CONVERSATIONAL"

        ra_res = engine.run_agent(greeting, groq_api_key="")
        assert ra_res["sources"] == []
        assert ra_res["execution_meta"]["execution_strategy"] == "Conversation"
        assert ra_res["execution_meta"]["retrieval_performed"] is False
        assert ra_res["execution_meta"]["citation_verification"] == "Not applicable"

        stream_evs = list(engine.run_adaptive_stream(greeting, groq_api_key=""))
        finals = [e for e in stream_evs if e.get("type") == "final"]
        assert len(finals) == 1
        assert finals[0]["sources"] == []
        assert finals[0]["execution_meta"]["execution_strategy"] == "Conversation"

    engine.search.assert_not_called()

    # Short Pega knowledge questions MUST NOT be treated as conversational
    for pega_q in ("What is SLA?", "Data Page?", "SLA?"):
        assert engine.detect_conversational_intent(pega_q) is None
        cls_pega = engine.classify_query_intent(pega_q)
        assert cls_pega["route"] == "DIRECT_RAG"


def test_ux_output_format_detection_clsa_prompts_and_streaming_metadata(isolated_env):
    """Verify format detection (TABLE/BULLETS/NUMBERED_STEPS/EXAMPLE/PARAGRAPH), CLSA prompts, and execution_meta."""
    from unittest.mock import MagicMock, patch
    from types import SimpleNamespace
    import rag_engine
    from agents.answer_generator import GROUNDED_SYSTEM_PROMPT
    from agents.research_agent import AGENT_SYSTEM_PROMPT

    assert rag_engine.detect_output_format("Compare Queue Processor vs Job Scheduler") == "TABLE"
    assert rag_engine.detect_output_format("Show Data Page scopes in a table") == "TABLE"
    assert rag_engine.detect_output_format("Give me step-by-step instructions to configure Connect-REST") == "NUMBERED_STEPS"
    assert rag_engine.detect_output_format("List the key points of Rule Resolution") == "BULLETS"
    assert rag_engine.detect_output_format("Give an example of Circumstancing") == "EXAMPLE"
    assert rag_engine.detect_output_format("What is PyWorkPage?") == "PARAGRAPH"

    # CLSA personality, comparison table, and selective follow-up directives present
    for prompt in (GROUNDED_SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT):
        assert "Lead System Architect (CLSA)" in prompt
        assert "Markdown comparison table" in prompt
        assert "follow-up question" in prompt

    engine = isolated_env["engine"]
    chunks = [
        {
            "source": "CLSA_Guide.pdf",
            "page": "18",
            "section": "Async Processing",
            "score": 90.0,
            "document": "Queue Processors handle item-based async work while Job Schedulers run on a cron schedule.",
        }
    ]
    engine.search = MagicMock(return_value=chunks)

    captured_prompts = []

    def _fake_create(**kwargs):
        captured_prompts.append(kwargs["messages"][-1]["content"])
        return iter([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="| Feature | QP |\n|---|---|\n"), finish_reason=None)]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=""), finish_reason="stop")]),
        ])

    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="openai/gpt-oss-120b")]
        )
        mock_client.chat.completions.create.side_effect = _fake_create
        mock_groq_cls.return_value = mock_client

        events = list(
            engine.run_adaptive_stream(
                "Compare Queue Processor vs Job Scheduler",
                groq_api_key="gsk_test",
            )
        )
        finals = [e for e in events if e.get("type") == "final"]
        assert len(finals) == 1
        assert "REQUESTED OUTPUT FORMAT: TABLE" in captured_prompts[0]
        meta = finals[0]["execution_meta"]
        assert meta["execution_strategy"] == "Direct RAG"
        assert meta["output_format"] == "TABLE"
        assert meta["evidence_status"] == "Sufficient"
        assert isinstance(meta["citation_verification"], dict)
        assert len(meta["documents_consulted"]) == 1
        assert rag_engine.ENABLE_ADAPTIVE_ROUTING is False


def test_defect_stage2_multi_domain_investigation_and_resolved_sufficiency(isolated_env):
    """Verify multi-domain investigation routing, off-topic top-chunk escalation, and resolved-query sufficiency."""
    from unittest.mock import MagicMock, patch
    from types import SimpleNamespace
    import rag_engine

    engine = isolated_env["engine"]

    # 1. Test 9: Multi-domain relationship investigation routes to AUTONOMOUS_AGENT
    res_rel = engine.classify_query_intent(
        "Investigate the relationship between SLA, Queue Processor and Job Scheduler in Pega."
    )
    assert res_rel["route"] == "AUTONOMOUS_AGENT"
    assert res_rel["tier"] == "deterministic"

    # Simple definitions & straightforward comparisons stay on DIRECT_RAG
    assert engine.classify_query_intent("What is SLA in Pega?")["route"] == "DIRECT_RAG"
    assert engine.classify_query_intent("Compare Queue Processor vs Job Scheduler")["route"] == "DIRECT_RAG"

    # 2. Off-topic #1 chunk (high score, 0 query-term overlap) fails evidence sufficiency
    off_topic_chunks = [
        {
            "source": "02_work-assignment.pdf",
            "page": "18",
            "score": 98.0,
            "document": "WorkBasket.GetNextWork routes human assignments to operator worklists.",
        },
        {
            "source": "02_designing-specialization.pdf",
            "page": "8",
            "score": 97.1,
            "document": "Circumstancing specializes rules based on property or date conditions.",
        },
    ]
    ev_off = rag_engine.evaluate_evidence_sufficiency(
        "What are the advantages of using circumstancing?", off_topic_chunks, intent="SIMPLE_LOOKUP"
    )
    assert ev_off["sufficient"] is False
    assert "off-topic" in ev_off["reason"].lower()

    # 3. Resolved follow-up query ('What about its caching strategy?' after Data Pages)
    #    rejects generic platform-cache chunks that lack 'data'/'pages' and triggers bounded escalation
    platform_cache_chunks = [
        {
            "source": "PlatformCache.pdf",
            "page": "4",
            "score": 94.0,
            "document": "Rule Cache, Conclusion Cache, and VTable cache rule metadata in memory.",
        }
    ]
    engine.search = MagicMock(return_value=platform_cache_chunks)
    engine.tool_search_curriculum = MagicMock(
        return_value={
            "results": [
                {
                    "document": "07_reporting-data-architecture.pdf",
                    "page": 25,
                    "section": "Data Page Caching",
                    "relevance_score": "99.0%",
                    "excerpt": "Data Pages support Thread, Requestor, and Node scopes with refresh strategies.",
                }
            ]
        }
    )

    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(id="openai/gpt-oss-120b"),
                SimpleNamespace(id="qwen/qwen3.8-27b"),
            ]
        )
        tool_call = SimpleNamespace(
            id="tc_dp_cache",
            function=SimpleNamespace(name="search_curriculum", arguments='{"query":"Data Page caching strategy","top_k":3}'),
        )
        mock_client.chat.completions.create.side_effect = [
            # LLM router returns DIRECT_RAG for follow-up
            SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"intent":"TECHNICAL_EXPLANATION","route":"DIRECT_RAG","reason":"follow-up"}'))],
                usage=SimpleNamespace(total_tokens=40),
            ),
            # Escalated Agent tool call + final answer
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="", tool_calls=[tool_call]))]),
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="Data Pages use Thread, Requestor, and Node scopes [07_reporting-data-architecture.pdf, Page 25, Section: Data Page Caching].", tool_calls=None))]),
        ]
        mock_groq_cls.return_value = mock_client

        res_fu = engine.run_adaptive(
            "What about its caching strategy?",
            groq_api_key="gsk_test",
            chat_history=[
                {"role": "user", "content": "Explain Data Pages in Pega."},
                {"role": "assistant", "content": "Data Pages load and cache data on demand."},
            ],
        )
        assert res_fu["routing"]["escalated"] is True
        assert res_fu["routing"]["escalation_count"] == 1
        assert "Data Pages" in res_fu["answer"]


def test_defect_stage3_grounding_prompts_circumstancing_and_truncation(isolated_env):
    """Verify Stage 3 circumstancing prioritization, Queue Processor/Work Queue prompt guards, and Markdown trimming."""
    from unittest.mock import MagicMock
    import rag_engine
    from agents.answer_generator import GROUNDED_SYSTEM_PROMPT, trim_to_last_complete_sentence
    from agents.research_agent import AGENT_SYSTEM_PROMPT, _retrieve_direct_rag_chunks

    # 1. Prompts distinguish Queue Processor vs Work Queue, structure Data Page troubleshooting,
    #    and forbid unsupported general-knowledge padding on insufficient evidence
    for prompt in (GROUNDED_SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT):
        assert "Queue Processor" in prompt and "Work Queue" in prompt
        assert "Thread" in prompt and "Requestor" in prompt and "Node" in prompt
        assert "do NOT pad with speculative explanations" in prompt

    # 2. Circumstancing prioritization demotes/excludes unrelated Work Queue chunks
    engine = isolated_env["engine"]
    engine.search = MagicMock(
        return_value=[
            {"source": "02_work-assignment.pdf", "page": "18", "score": 98.0, "document": "WorkBasket.GetNextWork routes human work queue items."},
            {"source": "02_designing-specialization.pdf", "page": "6", "score": 95.0, "document": "Circumstancing specializes rules without duplicating class hierarchies."},
            {"source": "02_designing-specialization.pdf", "page": "8", "score": 94.0, "document": "Circumstanced versions support property and date-based specialization."},
        ]
    )
    prioritized = _retrieve_direct_rag_chunks(
        engine=engine,
        query="What are the advantages of using circumstancing?",
        chat_history=None,
        top_k=4,
        section_filter=None,
        use_reranker=True,
        dense_weight=0.5,
    )
    assert all("02_designing-specialization.pdf" in c["source"] for c in prioritized)

    # 3. Markdown list & table preservation during truncation trimming
    table_truncated = (
        "| Feature | Queue Processor | Job Scheduler |\n"
        "| --- | --- | --- |\n"
        "| Trigger | Item queued | Cron schedule |\n"
        "| Retries | Built-in broken queue | No automatic"
    )
    trimmed_table = trim_to_last_complete_sentence(table_truncated)
    assert trimmed_table.endswith("| Cron schedule |")
    assert "No automatic" not in trimmed_table

    list_truncated = (
        "Key Data Page checks:\n"
        "- Scope: Thread, Requestor, or Node\n"
        "- Refresh strategy: Reload once per interaction\n"
        "- Invalidation: Flush on"
    )
    trimmed_list = trim_to_last_complete_sentence(list_truncated)
    assert trimmed_list.endswith("- Refresh strategy: Reload once per interaction")
    assert "Flush on" not in trimmed_list


def test_defect_stage4_capability_fast_path_and_citation_presentation(isolated_env):
    """Verify Stage 4 capability fast path, Pega-question preservation, and clean UI citation separation."""
    from unittest.mock import MagicMock
    import rag_engine
    from ui.chat_panel import format_citation_badges_html

    engine = isolated_env["engine"]
    engine.search = MagicMock(return_value=[])

    # Capability questions bypass RAG and citations
    for cap_q in (
        "What can you help me with?",
        "What can you help with?",
        "How can you help?",
        "How can you help me?",
    ):
        cls_res = engine.classify_query_intent(cap_q)
        assert cls_res["intent"] == "CONVERSATIONAL"
        assert cls_res["route"] == "CONVERSATIONAL"

        evs = list(engine.run_adaptive_stream(cap_q, groq_api_key=""))
        finals = [e for e in evs if e.get("type") == "final"]
        assert len(finals) == 1
        assert finals[0]["sources"] == []
        assert finals[0]["execution_meta"]["execution_strategy"] == "Conversation"
        assert finals[0]["execution_meta"]["retrieval_performed"] is False

    engine.search.assert_not_called()

    # Technical Pega questions with capability phrasing still route to RAG
    assert engine.detect_conversational_intent("How can you help me with Data Pages?") is None
    assert engine.classify_query_intent("How can you help me with Data Pages?")["route"] != "CONVERSATIONAL"

    # Citation badges include visual spacing separator
    badges_html = format_citation_badges_html([{"source": "CLSA.pdf", "page": "4", "section": "Data Pages"}])
    assert "margin-top:" in badges_html
    assert "CLSA.pdf" in badges_html







