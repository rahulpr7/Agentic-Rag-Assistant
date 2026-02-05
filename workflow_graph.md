---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	handle_memories(handle_memories)
	summarize_messages(summarize_messages)
	answer_or_retrieve(answer_or_retrieve)
	retrieve(retrieve)
	score_documents(score_documents)
	rewrite_query(rewrite_query)
	generate_answer(generate_answer)
	__end__([<p>__end__</p>]):::last
	__start__ --> handle_memories;
	__start__ --> summarize_messages;
	answer_or_retrieve -.-> __end__;
	answer_or_retrieve -.-> retrieve;
	generate_answer -.-> __end__;
	handle_memories -.-> answer_or_retrieve;
	retrieve --> score_documents;
	rewrite_query -.-> retrieve;
	score_documents -.-> generate_answer;
	score_documents -.-> rewrite_query;
	summarize_messages --> answer_or_retrieve;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
