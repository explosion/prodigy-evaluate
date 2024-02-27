from typing import Dict, Tuple, Callable, Iterable, List, Union, Sequence

from functools import partial

# spacy imports
from spacy import Language
from spacy.training.example import Example
from spacy.tokens.doc import SetEntsDefault
from spacy.tokens import Doc
from spacy.cli._util import string_to_list

# prodigy imports
from prodigy.errors import RecipeError
from prodigy.util import (
    EVAL_PREFIX,
    NER_DEFAULT_INCORRECT_KEY,
    SPANCAT_DEFAULT_KEY,
    COREF_DEFAULT_PREFIX,
)
from prodigy.recipes.data_utils import (
    get_datasets_from_cli,
    create_ner_reader,
    create_textcat_reader,
    create_parser_reader,
    create_tagger_reader,
    create_senter_reader,
    create_spancat_reader,
    create_coref_reader,
    create_merged_corpus,
)


def merge_corpus(
    nlp: Language,
    pipes: Dict[str, Tuple[List[str], List[str]]],
    *,
    eval_split: float = 0.0,
) -> Dict[str, Callable[[Language], Iterable[Example]]]:
    default_fill = SetEntsDefault.outside
    # determine relevant span keys from the nlp pipeline components
    ner_incorrect_key = NER_DEFAULT_INCORRECT_KEY
    ner_pipe = None
    if "ner" in nlp.pipe_names:
        ner_pipe = nlp.get_pipe("ner")
    elif "beam_ner" in nlp.pipe_names:
        ner_pipe = nlp.get_pipe("beam_ner")
    if ner_pipe and hasattr(ner_pipe, "incorrect_spans_key"):
        ner_incorrect_key = ner_pipe.incorrect_spans_key
    spans_key = SPANCAT_DEFAULT_KEY
    if "spancat" in nlp.pipe_names:
        spans_key = nlp.get_pipe("spancat").key
    coref_prefix = COREF_DEFAULT_PREFIX
    if "coref" in nlp.pipe_names:
        coref_prefix = nlp.get_pipe("coref").prefix

    reader_factories = {
        "ner": partial(
            create_ner_reader,
            default_fill=default_fill,
            incorrect_key=ner_incorrect_key,
        ),
        "textcat": partial(create_textcat_reader, exclusive=True),
        "textcat_multilabel": partial(create_textcat_reader, exclusive=False),
        "parser": create_parser_reader,
        "tagger": create_tagger_reader,
        "senter": create_senter_reader,
        "spancat": partial(
            create_spancat_reader,
            spans_key=spans_key,
        ),
        "experimental_coref": partial(create_coref_reader, coref_prefix=coref_prefix),
    }
    readers = {}
    for pipe, (train_sets, eval_sets) in pipes.items():
        if pipe in reader_factories:
            readers[pipe] = reader_factories[pipe](train_sets, eval_sets)
    corpus = create_merged_corpus(**readers, eval_split=eval_split)
    return corpus


def merge_data(
    nlp: Language,
    ner_datasets: Union[str, Sequence[str]] = tuple(),
    textcat_datasets: Union[str, Sequence[str]] = tuple(),
    textcat_multilabel_datasets: Union[str, Sequence[str]] = tuple(),
    tagger_datasets: Union[str, Sequence[str]] = tuple(),
    senter_datasets: Union[str, Sequence[str]] = tuple(),
    parser_datasets: Union[str, Sequence[str]] = tuple(),
    spancat_datasets: Union[str, Sequence[str]] = tuple(),
    coref_datasets: Union[str, Sequence[str]] = tuple(),
    *,
    eval_split: float = 0.0,
) -> Tuple[List[Doc], List[Doc], Dict[str, Tuple[List[str], List[str]]]]:
    pipes = get_datasets_from_cli(
        ner_datasets,
        textcat_datasets,
        textcat_multilabel_datasets,
        tagger_datasets,
        senter_datasets,
        parser_datasets,
        spancat_datasets,
        coref_datasets,
    )
    merged_corpus = merge_corpus(nlp=nlp, pipes=pipes, eval_split=eval_split)

    train_docs = [eg.reference for eg in merged_corpus["train"](nlp)]
    dev_docs = [eg.reference for eg in merged_corpus["dev"](nlp)]
    return train_docs, dev_docs, pipes


def get_datasets_from_cli_eval(
    ner: Union[Sequence[str], str],
    textcat: Union[Sequence[str], str],
    textcat_multilabel: Union[Sequence[str], str],
    tagger: Union[Sequence[str], str],
    senter: Union[Sequence[str], str],
    parser: Union[Sequence[str], str],
    spancat: Union[Sequence[str], str],
    coref: Union[Sequence[str], str],
) -> Dict[str, List[str]]:
    """Load evaluation sets based on list specified on the CLI.
    Takes care of converting a comma-separated string of names if needed, ensures
    that data is available and supports evaluation sets with the eval: prefix.

    *args (Union[Sequence[str], str]): The dataset names.
    RETURNS (Dict[str, Tuple[List[str], List[str]]]): The loaded examples, keyed
        by component name, with a value of a list of evaluation examples.
    """
    set_names = {
        "ner": ner,
        "tagger": tagger,
        "senter": senter,
        "parser": parser,
        "spancat": spancat,
        "experimental_coref": coref,
        "textcat": textcat,
        "textcat_multilabel": textcat_multilabel,
    }
    pipes = {}
    for name, value in set_names.items():
        if isinstance(value, str):
            value = string_to_list(value) if value else []
        eval_sets = []
        for n in value:
            if n.startswith(EVAL_PREFIX):
                eval_sets.append(n[len(EVAL_PREFIX) :])
            else:
                eval_sets.append(n)
        pipes[name] = eval_sets
    if not pipes:
        raise RecipeError(
            "You need to specify at least one eval dataset using one of the CLI options",
            ", ".join(f"--{c}" for c in set_names),
        )
    return pipes
