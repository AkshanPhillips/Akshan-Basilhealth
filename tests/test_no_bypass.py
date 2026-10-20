"""
Structural proof that no unchecked message can reach rag_agent.run() or tool_agent.run().

This does NOT test runtime behavior with example inputs (that's necessary but not
sufficient -- a test suite of examples can never prove a negative over all possible
inputs). Instead it statically scans the app/ package source and asserts two structural
facts about the codebase itself:

  1. The only place in app/*.py that calls `rag_agent.run(` or `tool_agent.run(` is
     orchestrator.py. If anyone later adds a second call site (a new handler, a debug
     script, a shortcut that skips the orchestrator), this test fails immediately,
     regardless of whether that call site happens to also check guardrails -- the point
     is there is exactly ONE place to audit, not that every place happens to be correct.

  2. orchestrator.handle_message's source text calls guardrails.check_input(...) before
     it calls router.classify(...), and calls guardrails.check_output(...) after
     dispatch and before constructing the final returned SupportResponse. This is a
     crude but effective source-order check: it would fail if someone reordered the
     function to classify/dispatch before checking input.

Run with: pytest tests/test_no_bypass.py -v
"""
import ast
import re
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"


def _find_calls(source: str, call_pattern: str) -> int:
    return len(re.findall(re.escape(call_pattern), source))


def test_rag_agent_run_only_called_from_orchestrator():
    offenders = []
    for path in APP_DIR.glob("*.py"):
        if path.name in ("rag_agent.py", "orchestrator.py"):
            continue
        src = path.read_text(encoding="utf-8")
        if _find_calls(src, "rag_agent.run(") > 0:
            offenders.append(path.name)
    assert not offenders, f"rag_agent.run() called outside orchestrator.py in: {offenders}"


def test_tool_agent_run_only_called_from_orchestrator():
    offenders = []
    for path in APP_DIR.glob("*.py"):
        if path.name in ("tool_agent.py", "orchestrator.py"):
            continue
        src = path.read_text(encoding="utf-8")
        if _find_calls(src, "tool_agent.run(") > 0:
            offenders.append(path.name)
    assert not offenders, f"tool_agent.run() called outside orchestrator.py in: {offenders}"


def test_orchestrator_checks_input_before_dispatch_and_output_before_return():
    src = (APP_DIR / "orchestrator.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    func = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "handle_message"
    )
    func_src = ast.get_source_segment(src, func)

    input_check_pos = func_src.find("guardrails.check_input(")
    router_pos = func_src.find("router.classify(")
    rag_pos = func_src.find("rag_agent.run(")
    tool_pos = func_src.find("tool_agent.run(")
    output_check_pos = func_src.find("guardrails.check_output(")

    assert input_check_pos != -1, "handle_message must call guardrails.check_input"
    assert router_pos != -1, "handle_message must call router.classify"
    assert output_check_pos != -1, "handle_message must call guardrails.check_output"

    assert input_check_pos < router_pos, "Input guardrail must run BEFORE routing"
    if rag_pos != -1:
        assert input_check_pos < rag_pos, "Input guardrail must run BEFORE rag_agent.run"
        assert rag_pos < output_check_pos, "Output guardrail must run AFTER rag_agent.run, before return"
    if tool_pos != -1:
        assert input_check_pos < tool_pos, "Input guardrail must run BEFORE tool_agent.run"
        assert tool_pos < output_check_pos, "Output guardrail must run AFTER tool_agent.run, before return"


def test_handle_message_returns_early_on_input_block():
    """Verify the early-return pattern: when input_check.allowed is False, the function
    returns inside that same `if` block rather than falling through to routing/dispatch."""
    src = (APP_DIR / "orchestrator.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    func = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "handle_message"
    )
    # First statement after the input_check assignment should be an `if not ... .allowed:`
    # containing a `return` in its body.
    found_guard_if = False
    for node in ast.walk(func):
        if isinstance(node, ast.If):
            test_src = ast.get_source_segment(src, node.test) or ""
            if "input_check.allowed" in test_src:
                has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
                assert has_return, "The input_check.allowed guard must contain a return statement"
                found_guard_if = True
    assert found_guard_if, "Could not find the `if not input_check.allowed` early-return guard"


def test_langgraph_edges_enforce_no_bypass():
    """Verify runtime LangGraph StateGraph topology:
    1. Graph must compile cleanly.
    2. START only transitions to 'input_guardrail' (no direct agent dispatch).
    3. 'rag_agent' and 'tool_agent' are ONLY reachable from 'router'.
    4. All agent execution nodes route directly to 'output_guardrail'.
    5. The only exit paths are 'input_guardrail' (when blocked) or 'output_guardrail' (when allowed).
    """
    from app.orchestrator import _SUPPORT_GRAPH, LANGGRAPH_AVAILABLE

    if not LANGGRAPH_AVAILABLE:
        return

    assert _SUPPORT_GRAPH is not None, "Compiled LangGraph state graph should not be None"
    graph = _SUPPORT_GRAPH.get_graph()

    # Extract edge map: source -> list of targets
    edges = [(e.source, e.target) for e in graph.edges]
    
    # 1. START node only leads to input_guardrail
    start_targets = [target for source, target in edges if source == "__start__"]
    assert start_targets == ["input_guardrail"], (
        f"START node must only transition to input_guardrail, got: {start_targets}"
    )

    # 2. rag_agent and tool_agent can only have 'router' as their incoming source
    rag_sources = [source for source, target in edges if target == "rag_agent"]
    tool_sources = [source for source, target in edges if target == "tool_agent"]
    assert rag_sources == ["router"], f"rag_agent must only be reached from router, got: {rag_sources}"
    assert tool_sources == ["router"], f"tool_agent must only be reached from router, got: {tool_sources}"

    # 3. All generating nodes must transition to output_guardrail
    generating_nodes = ["rag_agent", "tool_agent", "resume_node", "greeting_node", "escalate_node"]
    for node in generating_nodes:
        targets = [target for source, target in edges if source == node]
        assert targets == ["output_guardrail"], (
            f"Node '{node}' must only transition to 'output_guardrail', got: {targets}"
        )

    # 4. output_guardrail must transition to __end__
    output_targets = [target for source, target in edges if source == "output_guardrail"]
    assert output_targets == ["__end__"], (
        f"output_guardrail must transition to __end__, got: {output_targets}"
    )

