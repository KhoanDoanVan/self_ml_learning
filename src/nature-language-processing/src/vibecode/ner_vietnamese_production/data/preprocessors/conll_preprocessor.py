
from typing import List, Dict, Tuple



class CoNLLPreprocessor:
    """Preprocessor for CoNLL format data."""

    @staticmethod
    def read_conll_file(file_path: str) -> List[Tuple[List[str], List[str]]]:
        """
        Read CoNLL format file
        """
        sentences = []
        current_tokens = []
        current_labels = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if not line:
                    # Empty line indicates sentence boundary
                    if current_tokens:
                        sentences.append((current_tokens, current_labels))
                        current_tokens = []
                        current_labels = []
                elif line.startswith('#'):
                    # Skip comments
                    continue
                else:
                    # Parse token and label
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[-1]
                        current_tokens.append(token)
                        current_labels.append(label)


        # Add last sentence
        if current_tokens:
            sentences.append((current_tokens, current_labels))

        return sentences
    

    @staticmethod
    def write_conll_file(
        file_path: str,
        sentences: List[Tuple[List[str], List[str]]]
    ) -> None:
        """
        Write data to CoNLL format file
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for tokens, labels in sentences:
                for token, label in zip(tokens, labels):
                    f.write(f"{token} {label}\n")
                f.write("\n")


    @staticmethod
    def validate_bio_tags(
        labels: List[str]
    ) -> bool:
        """
        Validate BIO tagging scheme
        """
        for i, label in enumerate(labels):
            if label.startswith('I-'):
                # I- tag must follow B- or I- of same type
                if i == 0:
                    return False
                prev_label = labels[i - 1]
                entity_type = label[2:]
                if prev_label == '0' or (
                    prev_label.startswith(('B-', 'I-')) and prev_label[2:] != entity_type
                ):
                    return False
        
        return True