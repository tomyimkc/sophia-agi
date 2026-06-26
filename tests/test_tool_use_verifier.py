from agent.tool_use.verifier import trace_passes, verify_call, verify_decision, verify_trace

def test_valid_check_claim():
    assert verify_call("check_claim", {"text": "Confucius wrote the Dao De Jing"}).passed

def test_malformed_rejected():
    assert not verify_call("check_claim", {"wrong": "x"}).passed

def test_over_call():
    assert not verify_decision(True, {"decision": "answer_direct"}).passed

def test_full_trace():
    label={"decision":"call","tool_id":"check_claim","gold_answer":"No, Confucius did not write the Dao De Jing."}
    tc=[{"name":"check_claim","arguments":{"text":"Confucius wrote the Dao De Jing"}}]
    assert trace_passes(verify_trace(answer="No, Confucius did not write the Dao De Jing.", tool_calls=tc, label=label, tool_results=[{"result":{"passed":False}}], trace_turns=[{},{"final":True}]))

def test_gbnf():
    from agent.structured_output import schema_to_gbnf
    from provenance_bench.local_agent import TOOL_SCHEMAS
    assert "root ::=" in schema_to_gbnf(TOOL_SCHEMAS[0]["function"]["parameters"])
