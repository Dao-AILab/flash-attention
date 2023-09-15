# Copied from https://github.com/stanford-crfm/mistral/blob/main/src/corpora/detokenization.py
# Which was originally from https://github.com/NVIDIA/Megatron-LM/blob/aed2f75e209e525c842aec7c044af7acae2a4614/tasks/zeroshot_gpt/detokenizer.py

"""
Handle detokenization for different dataset for zero-shot LM evaluation.
"""
import re


def wikitext_detokenize(string: str) -> str:
    """
    Wikitext is whitespace tokenized and we remove these whitespaces.
    Taken from https://github.com/NVIDIA/Megatron-LM/blob/main/tasks/zeroshot_gpt2/detokenizer.py
    """
    # Contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)

    # Number Separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")

    # Punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")

    # Double Brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)

    # Miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


# Set Registry for Various Datasets
DATASET_TOKENIZATION_REGISTRY = {"wikitext": wikitext_detokenize}
