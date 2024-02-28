import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import spacy
import srsly
from radicli import Arg
from spacy.cli.evaluate import handle_scores_per_type
from spacy.language import Language
from spacy.training import offsets_to_biluo_tags
from spacy.training.example import Example

from prodigy.core import recipe
from prodigy.errors import RecipeError
from prodigy.util import SPANCAT_DEFAULT_KEY, msg
from prodigy.recipes.train import RECIPE_ARGS, set_log_level, setup_gpu

from .utils import get_datasets_from_cli_eval, merge_corpus

# additional imports
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from nervaluate import Evaluator


@recipe(
    "evaluate.evaluate",
    # fmt: off
    model=Arg(help="Name or path of model to evaluate"),
    ner=RECIPE_ARGS["ner"],
    textcat=RECIPE_ARGS["textcat"],
    textcat_multilabel=RECIPE_ARGS["textcat_multilabel"],
    tagger=RECIPE_ARGS["tagger"],
    senter=RECIPE_ARGS["senter"],
    parser=RECIPE_ARGS["parser"],
    spancat=RECIPE_ARGS["spancat"],
    coref=RECIPE_ARGS["coref"],
    label_stats=Arg("--label-stats", "-LS", help="Show per-label scores"),
    gpu_id=RECIPE_ARGS["gpu_id"],
    verbose=RECIPE_ARGS["verbose"],
    silent=RECIPE_ARGS["silent"],
    cf_matrix = Arg("--confusion-matrix", "-CF",  help="Show confusion matrix for the specified component"),
    cf_path = Arg("--cf-path", "-CP", help="Path to save the confusion matrix array"),
    spans_key=Arg("--spans-key", help="Optional spans key to evaluate if spancat component is used."),
    # fmt: on
)
def evaluate(
    model: Union[str, Path],
    ner: Sequence[str] = tuple(),
    textcat: Sequence[str] = tuple(),
    textcat_multilabel: Sequence[str] = tuple(),
    tagger: Sequence[str] = tuple(),
    senter: Sequence[str] = tuple(),
    parser: Sequence[str] = tuple(),
    spancat: Sequence[str] = tuple(),
    coref: Sequence[str] = tuple(),
    label_stats: bool = False,
    gpu_id: int = -1,
    verbose: bool = False,
    silent: bool = False,
    cf_matrix: bool = False,
    cf_path: Optional[Path] = None,
    spans_key: str = SPANCAT_DEFAULT_KEY,
) -> Dict[str, Any]:
    """Evaluate a spaCy pipeline on one or more datasets for different components.

    Per-component evaluation sets can be provided using the eval: prefix for consistency
    with the prodigy train command but are not required. All datasets are treated as
    eval datasets e.g. --ner my_train_dataset,eval:my_eval_dataset.

    This command takes care of merging all annotations on the same input data like the
    prodigy train command.

    You can also use the --label-stats flag to show per-label scores for NER and textcat
    components. This will show the precision, recall and F-score for each label.

    Finally, you can also use --confusion-matrix to show the confusion matrix for the
    specified component. This will only work for NER, textcat and textcat_multilabel components.

    prodigy evaluate en_core_web_sm --ner my_eval_dataset --label-stats --confusion-matrix
    """
    set_log_level(verbose=verbose, silent=silent)
    setup_gpu(gpu_id)
    nlp = spacy.load(model)

    pipes = get_datasets_from_cli_eval(
        ner,
        textcat,
        textcat_multilabel,
        tagger,
        senter,
        parser,
        spancat,
        coref,
    )
    pipe_key = [k for k in pipes if pipes.get(k)][0]

    compat_pipes = {
        pipe_name: ([], eval_sets) for pipe_name, eval_sets in pipes.items()
    }
    merged_corpus = merge_corpus(nlp, compat_pipes)
    dev_examples = merged_corpus["dev"](nlp)
    scores = nlp.evaluate(dev_examples)

    actual_labels, predicted_labels, labels = _get_cf_actual_predicted(
        nlp=nlp, dev_examples=dev_examples, pipe_key=pipe_key
    )
    labels_to_include = [l for l in labels if l != "O"]
    if pipe_key == "ner":
        actual_labels = [
            label.split("-")[-1] for sublist in actual_labels for label in sublist
        ]
        predicted_labels = [
            label.split("-")[-1] for sublist in predicted_labels for label in sublist
        ]

    if label_stats:
        _display_eval_results(
            scores=scores, spans_key=spans_key, silent=silent
        )

    if cf_matrix or cf_path:
        cfarray = confusion_matrix(
            actual_labels, predicted_labels, labels=labels_to_include, normalize="true"
        )
        if cf_matrix:
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cfarray, display_labels=labels_to_include
            )
            disp.plot()
            plt.show()
            msg.good(f"Confusion matrix displayed")

        if cf_path:
            if not cf_path.exists():
                os.makedirs(cf_path)

            full_cf_path = cf_path / "cf_array.json"
            srsly.write_json(
                full_cf_path,
                {
                    "cf_array": cfarray,
                    "labels": labels,
                },
            )
            msg.good(f"Confusion matrix array saved to {full_cf_path}")

    return scores


@recipe(
    "evaluate.evaluate-example",
    # fmt: off
    model=Arg(help="Path to model to evaluate"),
    ner=RECIPE_ARGS["ner"],
    textcat=RECIPE_ARGS["textcat"],
    textcat_multilabel=RECIPE_ARGS["textcat_multilabel"],
    tagger=RECIPE_ARGS["tagger"],
    senter=RECIPE_ARGS["senter"],
    parser=RECIPE_ARGS["parser"],
    spancat=RECIPE_ARGS["spancat"],
    coref=RECIPE_ARGS["coref"],
    gpu_id=RECIPE_ARGS["gpu_id"],
    verbose=RECIPE_ARGS["verbose"],
    silent=RECIPE_ARGS["silent"],
    metric=Arg("--metric", "-m", help="Metric to use for sorting examples"),
    n_results = Arg("--n-results", "-NR", help="Number of top examples to display"),
    output_path=Arg("--output-path", "-OP", help="Path to save the top examples and scores")
    # fmt: on
)
def evaluate_example(
    model: Union[str, Path],
    ner: Sequence[str] = tuple(),
    textcat: Sequence[str] = tuple(),
    textcat_multilabel: Sequence[str] = tuple(),
    tagger: Sequence[str] = tuple(),
    senter: Sequence[str] = tuple(),
    parser: Sequence[str] = tuple(),
    spancat: Sequence[str] = tuple(),
    coref: Sequence[str] = tuple(),
    gpu_id: int = -1,
    verbose: bool = False,
    silent: bool = False,
    metric: Optional[str] = None,
    n_results: int = 10,
    output_path: Optional[Path] = None,
):
    """Evaluate a spaCy pipeline on one or more datasets for different components
    on a per-example basis. Datasets are provided in the same per-component format
    as the prodigy evaluate command e.g. --ner my_eval_dataset_1,eval:my_eval_dataset_2.
    This command will run an evaluation on each example individually and then sort by the
    desired `--metric` argument.

    This is useful for debugging and understanding the easiest
    and hardest examples for your model.

    Example Usage:
        ```
        prodigy evaluate-example en_core_web_sm --ner my_eval_dataset --metric ents_f
        ```

    This will sort examples by lowest NER F-score.
    """
    if not metric:
        raise RecipeError(
            "You must pass a metric to sort examples via --metric argument. Refer to prodigy evaluate-example documentation for available metric types."
        )

    set_log_level(verbose=verbose, silent=silent)
    setup_gpu(gpu_id)
    nlp = spacy.load(model)

    pipes = get_datasets_from_cli_eval(
        ner,
        textcat,
        textcat_multilabel,
        tagger,
        senter,
        parser,
        spancat,
        coref,
    )
    compat_pipes = {
        pipe_name: ([], eval_sets) for pipe_name, eval_sets in pipes.items()
    }
    merged_corpus = merge_corpus(nlp, compat_pipes)
    dev_examples = merged_corpus["dev"](nlp)
    results: List[ScoredExample] = evaluate_each_example(nlp, dev_examples, metric)

    top_results: List[ScoredExample] = results[:n_results]

    if len(top_results) == 0:
        msg.fail(f"No examples found for the metric {metric}.", exits=1)
    avg_text_len = sum([len(ex.example.text) for ex in top_results]) / len(top_results)
    if avg_text_len > 100:
        msg.warn(
            f"Average # of characters of top examples is {round(avg_text_len, 2)}. This will not display well in the terminal. Consider saving the top examples to file with `--output-path` and investigating accordingly."
        )
    data = [
        (ex.example.text, round(ex.score, 2) if ex.score is not None else None)
        for ex in top_results
    ]
    headers = ["Example", metric]
    widths = (max([len(d[0]) for d in data]), 9)
    aligns = ("l", "l")

    msg.divider("Scored Examples")
    msg.table(data, header=headers, divider=True, widths=widths, aligns=aligns)

    if output_path:
        if not output_path.exists():
            os.makedirs(output_path)

        results_path = output_path / "hardest_examples.jsonl"

        results_jsonl = []
        for data in top_results:
            results_json = {
                "text": data.example.text,
                "meta": {"score": data.score, "metric": metric},
            }
            results_jsonl.append(results_json)

        srsly.write_jsonl(results_path, results_jsonl)
        msg.good(f"The examples with the lowest scores saved to {results_path}")
        msg.info(
            f"You can inspect the hardest examples by running: `prodigy RECIPE DATASET MODEL {results_path}`. See documentation for more details: https://prodi.gy/docs/recipes"
        )


@recipe(
    "evaluate.nervaluate",
    # fmt: off
    model=Arg(help="Path to model to evaluate"),
    ner=RECIPE_ARGS["ner"],
    gpu_id=RECIPE_ARGS["gpu_id"],
    verbose=RECIPE_ARGS["verbose"],
    silent=RECIPE_ARGS["silent"],
    # fmt: on
)
def evaluate_nervaluate(
    model: Union[str, Path],
    ner: Sequence[str],
    gpu_id: int = -1,
    verbose: bool = False,
    silent: bool = False,
):
    """
    Evaluate spaCy's NER component using nervaluate metrics. the `nervaluate` library
    provides full named-entity (i.e. not tag/token) evaluation metrics based on SemEval’13.

    For more information on these metric, see https://github.com/MantisAI/nervaluate.

    Example Usage:

        ```
        prodigy evaluate.nervaluate en_core_web_sm my_eval_dataset
        ```
    """
    set_log_level(verbose=verbose, silent=silent)
    setup_gpu(gpu_id)
    nlp = spacy.load(model)
    merged_corpus = merge_corpus(nlp, {"ner": ([], ner)})
    dev_examples = merged_corpus["dev"](nlp)
    actual_labels, predicted_labels, labels = _get_cf_actual_predicted(
        nlp=nlp, dev_examples=dev_examples, pipe_key="ner"
    )

    evaluator = Evaluator(actual_labels, predicted_labels, tags=labels, loader="list")
    ner_results, ner_results_by_tag = evaluator.evaluate()
    msg.divider("nervaluate NER metrics")
    msg.info(
        "Full named-entity (i.e., not tag/token) evaluation metrics based on SemEval’13. For more information on these metrics, see https://github.com/MantisAI/nervaluate"
    )
    msg.text("NER: Overall")
    _create_ner_table(ner_results)

    if not silent:
        for tag, tag_results in ner_results_by_tag.items():
            if tag != "O":
                msg.text(title=f"NER: {tag}")
                _create_ner_table(tag_results)

    return {"overall_results": ner_results, "results_by_tag": ner_results_by_tag}


@dataclass
class ScoredExample:
    example: Example
    score: Optional[float]
    scores: Dict[str, float]


def _get_score_for_metric(scores: Dict[str, float], metric: str) -> Union[float, None]:

    return scores.get(metric, None)


def evaluate_each_example(
    nlp: Language,
    dev_examples: Iterable[Example],
    metric: str,
    desc: bool = False,
    skip_none: bool = True,
) -> List[ScoredExample]:
    def sort_key(x: Tuple[Example, Dict[str, float]]) -> Union[float, int]:
        _, eval_scores = x
        res = _get_score_for_metric(eval_scores, metric)
        if res is None:
            res = 0
        if not isinstance(res, (float, int)):
            raise ValueError(f"Invalid metric to sort by: {metric}", res)
        return res

    per_example_scores = {}
    for example in dev_examples:
        scores = nlp.evaluate([example])
        res = _get_score_for_metric(scores, metric)
        if res is None and skip_none:
            continue
        per_example_scores[example] = scores

    sorted_per_example_scores = [
        ScoredExample(
            example=eg,
            score=_get_score_for_metric(example_scores, metric),
            scores=example_scores,
        )
        for eg, example_scores in sorted(
            per_example_scores.items(), key=sort_key, reverse=desc
        )
    ]
    return sorted_per_example_scores


def _display_eval_results(
    scores: Dict[str, Any],
    spans_key: str,
    silent: bool = False,
) -> None:
    metrics = {
        "TOK": "token_acc",
        "TAG": "tag_acc",
        "POS": "pos_acc",
        "MORPH": "morph_acc",
        "LEMMA": "lemma_acc",
        "UAS": "dep_uas",
        "LAS": "dep_las",
        "NER P": "ents_p",
        "NER R": "ents_r",
        "NER F": "ents_f",
        "TEXTCAT": "cats_score",
        "SENT P": "sents_p",
        "SENT R": "sents_r",
        "SENT F": "sents_f",
        "SPAN P": f"spans_{spans_key}_p",
        "SPAN R": f"spans_{spans_key}_r",
        "SPAN F": f"spans_{spans_key}_f",
        "SPEED": "speed",
    }
    results = {}
    data = {}
    for metric, key in metrics.items():
        if key in scores:
            if key == "cats_score":
                metric = metric + " (" + scores.get("cats_score_desc", "unk") + ")"
            if isinstance(scores[key], (int, float)):
                if key == "speed":
                    results[metric] = f"{scores[key]:.0f}"
                else:
                    results[metric] = f"{scores[key]*100:.2f}"
            else:
                results[metric] = "-"
            data[re.sub(r"[\s/]", "_", key.lower())] = scores[key]
    msg.table(results, title="Results")
    data = handle_scores_per_type(scores, data, spans_key=spans_key, silent=silent)


#### Confusion matrix functions ####


def _get_actual_labels(dev_examples: Iterable[Example], pipe_key: str) -> List[Any]:
    """Returns the actual labels for the specified component.

    Args:
        dev_examples (Iterable[Example]): List of examples
        pipe_key (str): Name of the component

    Returns:
        List[Any]: List of actual labels
    """
    actual_labels = []
    for ex in dev_examples:
        ref = ex.reference  # we have reference but we don't have predicted
        if pipe_key == "ner":
            ents = ex.get_aligned_ner()
            ents_clean = ["O" if x is None else x for x in ents]
            actual_labels.append(ents_clean)
        elif pipe_key == "textcat":
            text_labels = ref.cats
            most_likely_class = (
                max(text_labels, key=lambda k: text_labels[k])
                if text_labels != {}
                else "O"
            )
            actual_labels.append(most_likely_class)

    return actual_labels


def _get_predicted_labels(
    nlp: Language, dev_examples: Iterable[Example], pipe_key: str
) -> List[Any]:
    """Returns the predicted labels for the specified component.

    Args:
        nlp (Language): spaCy model
        dev_examples (Iterable[Example]): List of examples
        pipe_key (str): Name of the component

    Returns:
        List[Any]: List of predicted labels
    """

    texts = [eg.text for eg in dev_examples]
    pred_labels = []
    for eg in nlp.pipe(texts):
        if pipe_key == "ner":
            ents = [(ent.start_char, ent.end_char, ent.label_) for ent in eg.ents]
            biluo_tags = offsets_to_biluo_tags(eg, ents)
            pred_labels.append(biluo_tags)
        elif pipe_key == "textcat":
            text_labels = eg.cats
            most_likely_class = (
                max(text_labels, key=lambda k: text_labels[k])
                if text_labels != {}
                else "O"
            )
            pred_labels.append(most_likely_class)

    return pred_labels


def _get_cf_actual_predicted(
    nlp: Language, dev_examples: Iterable[Example], pipe_key: str
):
    """Returns the actual and predicted labels for the specified component.

    Args:
        nlp (Language): spaCy model
        dev_examples (Iterable[Example]): List of examples
        pipe_key (str): Name of the component

    Returns:
        Tuple[List[Any], List[Any], List[Any]]: Tuple containing actual labels, predicted labels and labels
    """
    if pipe_key not in ["ner", "textcat"]:
        msg.fail(f"Unsupported {pipe_key} component", exits=1)
    else:
        actual_labels = [label for label in _get_actual_labels(dev_examples, pipe_key)]
        predicted_labels = [
            label for label in _get_predicted_labels(nlp, dev_examples, pipe_key)
        ]
        if pipe_key == "textcat":
            labels = set(predicted_labels).union(set(actual_labels))
        elif pipe_key == "ner":
            actual_labels_flat = [
                label.split("-")[-1] for sublist in actual_labels for label in sublist
            ]
            predicted_labels_flat = [
                label.split("-")[-1]
                for sublist in predicted_labels
                for label in sublist
            ]
            labels = set(predicted_labels_flat).union(set(actual_labels_flat))

        return actual_labels, predicted_labels, list(labels)


def _create_ner_table(results: Dict[str, Dict[str, float]]):
    """Creates a table for NER results.

    Args:
        results (Dict[str, Dict[str, float]]): Dictionary containing NER results.
    """

    ner_metrics = [
        "correct",
        "incorrect",
        "partial",
        "missed",
        "spurious",
        "possible",
        "actual",
        "precision",
        "recall",
        "f1",
    ]
    headers = tuple(["Metric"] + [m.capitalize() for m in ner_metrics])

    metrics_formatted = []
    for eval_type, metrics in results.items():
        row = [eval_type.replace("_", " ").capitalize()]
        row.extend(
            [
                (
                    round(metrics.get(key, None), 2)
                    if metrics.get(key, None) is not None
                    else None
                )
                for key in ner_metrics
            ]
        )
        metrics_formatted.append(row)

    msg.table(metrics_formatted, header=headers, divider=True)
