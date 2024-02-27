from typing import Dict, Iterable, List

import pytest

from spacy.training import Example
import en_core_web_sm

from prodigy.types import TaskType
from prodigy.components.db import connect

from prodigy_evaluate import (
    _display_eval_results,
    _get_score_for_metric,
    evaluate,
    evaluate_example,
    _create_cf_array,
    _get_actual_labels,
    _get_predicted_labels,
    _get_cf_actual_predicted,
)


@pytest.fixture
def dataset() -> str:
    return "test_dataset"


@pytest.fixture
def spacy_model():
    return "en_core_web_sm"


@pytest.fixture
def nlp():
    return en_core_web_sm.load()


@pytest.fixture
def metric() -> str:
    return "ents_f"


@pytest.fixture
def data() -> Iterable[Dict]:
    return [
        {
            "text": "My name is Freya.",
            "_input_hash": 896529854,
            "_task_hash": -1486695581,
            "tokens": [
                {"text": "My", "start": 0, "end": 2, "id": 0, "ws": True},
                {"text": "name", "start": 3, "end": 7, "id": 1, "ws": True},
                {"text": "is", "start": 8, "end": 10, "id": 2, "ws": True},
                {"text": "Freya", "start": 11, "end": 16, "id": 3, "ws": True},
                {"text": ".", "start": 16, "end": 17, "id": 4, "ws": True},
            ],
            "_view_id": "ner_manual",
            "spans": [
                {
                    "start": 11,
                    "end": 16,
                    "token_start": 3,
                    "token_end": 3,
                    "label": "PERSON",
                }
            ],
            "answer": "accept",
            "_timestamp": 1707211049,
            "_annotator_id": "2024-02-06_10-17-19",
            "_session_id": "2024-02-06_10-17-19",
        },
        {
            "text": "My favorite city is London.",
            "_input_hash": -91551573,
            "_task_hash": -1162253049,
            "tokens": [
                {"text": "My", "start": 0, "end": 2, "id": 0, "ws": True},
                {"text": "favorite", "start": 3, "end": 11, "id": 1, "ws": True},
                {"text": "city", "start": 12, "end": 16, "id": 2, "ws": True},
                {"text": "is", "start": 17, "end": 19, "id": 3, "ws": True},
                {"text": "London", "start": 20, "end": 26, "id": 4, "ws": True},
                {"text": ".", "start": 26, "end": 27, "id": 5, "ws": False},
            ],
            "_view_id": "ner_manual",
            "spans": [
                {
                    "start": 20,
                    "end": 26,
                    "token_start": 4,
                    "token_end": 4,
                    "label": "GPE",
                }
            ],
            "answer": "accept",
            "_timestamp": 1707211053,
            "_annotator_id": "2024-02-06_10-17-19",
            "_session_id": "2024-02-06_10-17-19",
        },
        {
            "text": "I live in Berlin.",
            "_input_hash": -2101464790,
            "_task_hash": 1279282044,
            "tokens": [
                {"text": "I", "start": 0, "end": 1, "id": 0, "ws": True},
                {"text": "live", "start": 2, "end": 6, "id": 1, "ws": True},
                {"text": "in", "start": 7, "end": 9, "id": 2, "ws": True},
                {"text": "Berlin", "start": 10, "end": 16, "id": 3, "ws": True},
                {"text": ".", "start": 16, "end": 17, "id": 4, "ws": True},
            ],
            "_view_id": "ner_manual",
            "spans": [
                {
                    "start": 10,
                    "end": 16,
                    "token_start": 3,
                    "token_end": 3,
                    "label": "GPE",
                }
            ],
            "answer": "accept",
            "_timestamp": 1707211056,
            "_annotator_id": "2024-02-06_10-17-19",
            "_session_id": "2024-02-06_10-17-19",
        },
    ]


@pytest.fixture
def scores() -> Dict[str, float]:
    return {
        "ents_f": 0.9,
        "ents_p": 0.8,
        "ents_r": 0.7,
        "tags_acc": 0.6,
        "sents_p": 0.5,
        "sents_r": 0.4,
        "sents_f": 0.3,
    }


@pytest.fixture
def db(dataset: str, data: List[TaskType]):
    database = connect()
    database.add_dataset(dataset)
    database.add_examples(data, datasets=[dataset])
    return database


@pytest.fixture
def ner_examples(nlp):
    data = {
        "Apple Inc. is an American multinational technology company.": {
            "entities": [(0, 10, "ORG")]  # Span covering "Apple Inc."
        },
        "Musk is the CEO of Tesla, Inc.": {
            "entities": [
                (0, 4, "PERSON"),
                (19, 30, "ORG"),
            ]  # Spans covering "Musk" and "Tesla, Inc."
        },
    }
    examples = []
    for text, annot in data.items():
        examples.append(Example.from_dict(nlp.make_doc(text), annot))

    return examples


@pytest.fixture
def textcat_examples(nlp):
    data = {
        "SpaCy is an amazing library for NLP.": {"POSITIVE": 1.0, "NEGATIVE": 0.0},
        "I dislike rainy days.": {"POSITIVE": 0.0, "NEGATIVE": 1.0},
    }

    examples = []
    for text, annot in data.items():
        doc = nlp.make_doc(text)
        doc.cats = annot
        ref_doc = nlp.make_doc(text)
        ref_doc.cats = annot
        example = Example(doc, ref_doc)
        examples.append(example)

    return examples


######## evaluation tests ########


def test_evaluate_example(spacy_model, dataset, metric, db, capsys):

    evaluate_example(model=spacy_model, ner=dataset, metric=metric, n_results=5)

    captured = capsys.readouterr()

    assert "Scored Example" in captured.out

    db.drop_dataset(dataset)


def test_evaluate(spacy_model, dataset, db, capsys):

    results = evaluate(
        model=spacy_model,
        ner=dataset,
        label_stats=True,
        confusion_matrix=True,
    )

    captured = capsys.readouterr()

    assert "Confusion Matrix" in captured.out
    assert "P" in captured.out
    assert "R" in captured.out
    assert "F" in captured.out

    assert isinstance(results, dict)
    assert "token_acc" in results
    assert "token_p" in results
    assert results.get("token_p") == 1
    assert isinstance(results.get("ents_p"), float)
    assert results.get("speed") > 1

    db.drop_dataset(dataset)


def test_display_eval_results(scores, capsys):
    _display_eval_results(scores, "sc")
    captured = capsys.readouterr()

    assert "Results" in captured.out


def test_get_score_for_metric(scores, metric: str):
    res = _get_score_for_metric(scores, metric)

    assert isinstance(res, float)
    assert isinstance(scores, dict)
    assert isinstance(metric, str)
    assert metric is not None


##### confusion matrix tests #####


def test_get_actual_labels_ner(ner_examples):

    ner_labels = _get_actual_labels(ner_examples, "ner")
    assert isinstance(ner_labels, list)
    assert len(ner_labels) == 17
    assert all(isinstance(label, str) for label in ner_labels)
    assert "O" in ner_labels
    assert "ORG" in ner_labels
    assert "PERSON" in ner_labels


def test_get_actual_labels_textcat(textcat_examples):

    textcat_labels = _get_actual_labels(textcat_examples, "textcat")
    assert isinstance(textcat_labels, list)
    assert len(textcat_labels) == 2
    assert "POSITIVE" in textcat_labels
    assert "NEGATIVE" in textcat_labels
    assert all(isinstance(label, str) for label in textcat_labels)


# here we need a model as we're using one in _get_predicted_labels
# because nlp.evaluate does not create example.predicted values
def test_get_predicted_labels_ner(nlp, ner_examples):

    pred_ner_labels = _get_predicted_labels(nlp, ner_examples, "ner")
    assert isinstance(pred_ner_labels, list)
    assert len(pred_ner_labels) == 17
    assert "O" in pred_ner_labels


def test_get_cf_actual_predicted(nlp, ner_examples):

    actual, predicted, labels = _get_cf_actual_predicted(nlp, ner_examples, "ner")
    assert isinstance(actual, list)
    assert isinstance(predicted, list)
    assert isinstance(labels, list)
    assert "ORG" in labels
    assert "ORG" in actual
    assert "PERSON" in actual
    assert "ORG" in predicted


@pytest.mark.parametrize(
    "actual_labels, predicted_labels, labels, expected_output",
    [
        (
            ["O", "O", "SKILL"],
            ["O", "O", "SKILL"],
            ["SKILL"],
            ([[1.0]], ["SKILL"]),
        ),  # agreement ner
        (
            ["O", "O", "SKILL"],
            ["O", "SKILL", "O"],
            ["SKILL"],
            ([[0]], ["SKILL"]),
        ),  # disagreement ner
        (
            ["CAT", "CAT", "DOG"],
            ["CAT", "CAT", "DOG"],
            ["CAT", "DOG"],
            ([[1.0, 0.0], [0.0, 1.0]], ["CAT", "DOG"]),
        ),  # agreement textcat
        (
            [1, 1, 0],
            [1, 0, 1],
            [1, 0],
            ([[0.5, 0.5], [1.0, 0.0]], [1, 0]),
        ),  # disagreement textcat
    ],
)
def test_create_cf_array(actual_labels, predicted_labels, labels, expected_output):
    actual_matrix, actual_labels = _create_cf_array(
        actual_labels, predicted_labels, labels
    )
    expected_matrix, expected_labels = expected_output

    # Compare confusion matrices structure and values
    for i, row in enumerate(expected_matrix):
        for j, value in enumerate(row):
            assert actual_matrix[i][j] == pytest.approx(
                value
            ), f"Value mismatch at position [{i}][{j}]"

    # Compare labels
    assert actual_labels == expected_labels, "Labels mismatch"

    # Additional checks can remain the same
    assert len(actual_matrix) == len(expected_matrix), "Matrix size mismatch"
    # Assert that all the values are between 0 and 1
    assert all(
        0 <= x <= 1 for row in actual_matrix for x in row
    ), "Values in the matrix are not within the [0, 1] range"
