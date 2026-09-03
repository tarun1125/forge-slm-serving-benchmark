from forge.phase2.prompts import (
    _build_short,
    _find_database_block,
    _first_collection_line,
    _schema_only,
    _split_paragraphs,
)

SAMPLE_BATCH_PROMPT = """You are a MongoDB query expert.
When given a natural-language question and a database schema, you output ONLY the raw PyMongo query.

Schema (2 databases, 3 collection(s)):

car_1:
- car_names: { MakeId (int), Model (str), Make (str) }
- cars_data: { Id (int), MPG (str), Cylinders (int) }

world_1:
- country: { Code (str), Name (str), Population (int) }

Note: some numeric fields are stored as strings.

Rules:
1. Output ONLY the PyMongo expression.
2. Do NOT wrap in markdown."""


class TestSplitParagraphs:
    def test_splits_on_blank_lines(self):
        paragraphs = _split_paragraphs(SAMPLE_BATCH_PROMPT)
        assert len(paragraphs) == 6  # preamble, header, car_1, world_1, note, rules

    def test_first_paragraph_is_the_preamble(self):
        paragraphs = _split_paragraphs(SAMPLE_BATCH_PROMPT)
        assert paragraphs[0].startswith("You are a MongoDB query expert.")


class TestFindDatabaseBlock:
    def test_finds_an_existing_database(self):
        paragraphs = _split_paragraphs(SAMPLE_BATCH_PROMPT)
        block = _find_database_block(paragraphs, "car_1")
        assert block is not None
        assert block.startswith("car_1:")
        assert "car_names" in block
        assert "cars_data" in block

    def test_does_not_confuse_schema_header_with_a_database(self):
        # "Schema (2 databases, ...):" also ends in a colon — a database
        # name that's merely a PREFIX of that header text (unlike an exact
        # match, which legitimately would and should match) must not
        # false-match a naive substring/prefix search.
        paragraphs = _split_paragraphs(SAMPLE_BATCH_PROMPT)
        block = _find_database_block(paragraphs, "Schema")
        assert block is None

    def test_missing_database_returns_none(self):
        paragraphs = _split_paragraphs(SAMPLE_BATCH_PROMPT)
        assert _find_database_block(paragraphs, "nonexistent_db") is None


class TestFirstCollectionLine:
    def test_returns_requested_gold_collection(self):
        block = _find_database_block(_split_paragraphs(SAMPLE_BATCH_PROMPT), "car_1")
        line = _first_collection_line(block, "car_1.cars_data")
        assert line is not None
        assert "cars_data" in line
        assert "car_names" not in line

    def test_falls_back_to_first_line_when_gold_collection_not_given(self):
        block = _find_database_block(_split_paragraphs(SAMPLE_BATCH_PROMPT), "car_1")
        line = _first_collection_line(block, None)
        assert line is not None
        assert "car_names" in line  # the first collection line in the block

    def test_falls_back_to_first_line_when_gold_collection_not_found(self):
        block = _find_database_block(_split_paragraphs(SAMPLE_BATCH_PROMPT), "car_1")
        line = _first_collection_line(block, "car_1.nonexistent_collection")
        assert line is not None
        assert "car_names" in line


class TestSchemaOnly:
    def test_strips_preamble_header_and_footer(self):
        schema = _schema_only(SAMPLE_BATCH_PROMPT)
        assert "You are a MongoDB query expert" not in schema
        assert "Schema (2 databases" not in schema
        assert "Rules:" not in schema
        assert "Note:" not in schema

    def test_keeps_database_blocks(self):
        schema = _schema_only(SAMPLE_BATCH_PROMPT)
        assert "car_1:" in schema
        assert "world_1:" in schema
        assert "car_names" in schema


class TestBuildShort:
    def test_returns_none_for_a_join_question(self):
        case = {"gold_collections": ["car_1.car_names", "car_1.cars_data"], "question": "q"}
        paragraphs = _split_paragraphs(SAMPLE_BATCH_PROMPT)
        preamble = paragraphs[0]
        block = _find_database_block(paragraphs, "car_1")
        assert _build_short(case, preamble, block) is None

    def test_builds_a_single_collection_prompt_for_a_simple_question(self):
        case = {"gold_collections": ["car_1.cars_data"], "question": "How many cars?"}
        paragraphs = _split_paragraphs(SAMPLE_BATCH_PROMPT)
        preamble = paragraphs[0]
        block = _find_database_block(paragraphs, "car_1")
        result = _build_short(case, preamble, block)
        assert result is not None
        system_prompt, question = result
        assert "cars_data" in system_prompt
        assert "car_names" not in system_prompt  # only the gold collection, not the whole db
        assert question == "How many cars?"
