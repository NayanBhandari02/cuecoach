from cuecoach.rag.schemas import Chunk


def test_chunk_schema_validates():
    c = Chunk(
        chunk_id="doc1::s1::c1",
        doc_id="doc1",
        source="TestSource",
        title="Test Title",
        text="This is a test chunk.",
        topic="aiming",
        skill_level="beginner",
    )
    assert c.topic == "aiming"
    assert c.skill_level == "beginner"
