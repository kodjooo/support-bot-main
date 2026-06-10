from app.ai.vector_client import _parse_context


def test_parse_context_reads_structured_results():
    chunks = _parse_context(
        {
            "results": [
                {
                    "text": "Строка удержаний раскрывается в Дэшборде.",
                    "score": 1.2,
                    "distance": 0.4,
                    "metadata": {"title": "Дэшборд / Удержания"},
                    "matched_terms": ["удержаний"],
                },
            ],
        },
    )

    assert chunks[0].text == "Строка удержаний раскрывается в Дэшборде."
    assert chunks[0].score == 1.2
    assert chunks[0].metadata == {"title": "Дэшборд / Удержания"}
    assert "Источник: Дэшборд / Удержания" in chunks[0].to_prompt_text()


def test_parse_context_keeps_legacy_chunks():
    chunks = _parse_context({"chunks": ["старый формат"]})

    assert chunks[0].text == "старый формат"
    assert chunks[0].to_prompt_text().endswith("Фрагмент: старый формат")
