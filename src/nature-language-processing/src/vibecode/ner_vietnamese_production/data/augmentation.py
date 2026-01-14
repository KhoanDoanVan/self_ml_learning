import random
from typing import List, Tuple
from utils.constants import ENTITY_TYPES



class NERDataAugmenter:

    def __init__(self, seed: int = 42):
        random.seed(seed)


    def synonym_replacement(
            self,
            tokens: List[str],
            labels: List[str],
            n: int = 1
    ) -> Tuple[List[str], List[str]]:
        """
        Replace non-entity words with synonyms
        """

        # Vietnamese synonym dictionary (simplified example)
        synonyms = {
            'lớn': ['to', 'khổng lồ', 'rộng'],
            'nhỏ': ['bé', 'nhỏ bé', 'tí hon'],
            'đẹp': ['xinh', 'đẹp đẽ', 'lung linh'],
            'xấu': ['tệ', 'dở', 'khó coi']
        }

        new_tokens = tokens.copy()
        non_entity_indices = [
            i for i, label in enumerate(labels)
            if label == '0'
        ]

        if not non_entity_indices:
            return new_tokens, labels
        
        # Replace n random non-entity words
        replace_indices = random.sample(
            non_entity_indices,
            min(n, len(non_entity_indices))
        )

        for idx in replace_indices:
            token = tokens[idx].lower()
            if token in synonyms:
                new_tokens[idx] = random.choice(synonyms[token])

        return new_tokens, labels
    

    def random_deletion(
            self,
            tokens: List[str],
            labels: List[str],
            p: float = 0.1
    ) -> Tuple[List[str], List[str]]:
        """
        Randomly delete non-entity words
        """
        if len(tokens) == 1:
            return tokens, labels
        
        new_tokens = []
        new_labels = []

        for token, label in zip(tokens, labels):
            # Keep entity words and randomly keep non-entity words
            if label != '0' or random.random() > p:
                new_tokens.append(token)
                new_labels.append(label)

        # Ensure at least one word remains
        if not new_tokens:
            idx = random.randint(0, len(tokens) - 1)
            new_tokens = [tokens[idx]]
            new_labels = [labels[idx]]

        return new_tokens, new_labels
    

    def random_swap(
            self,
            tokens: List[str],
            labels: List[str],
            n: int = 1
    ) -> Tuple[List[str], List[str]]:
        """
        Randomly swap non-entity words
        """
        new_tokens = tokens.copy()
        new_labels = labels.copy()

        non_entity_indices = [
            i for i, label in enumerate(labels)
            if label == '0'
        ]

        if len(non_entity_indices) < 2:
            return new_tokens, new_labels
        
        for _ in range(n):
            idx1, idx2 = random.sample(non_entity_indices, 2)
            new_tokens[idx1], new_tokens[idx2] = new_tokens[idx2], new_tokens[idx1]
            new_labels[idx1], new_labels[idx2] = new_labels[idx2], new_labels[idx1]

        return new_tokens, new_labels
    

    def entity_replacement(
            self,
            tokens: List[str],
            labels: List[str],
            entity_pool: dict
    ) -> Tuple[List[str], List[str]]:
        """
        Replace entities with similar entities from pool
        """
        new_tokens = tokens.copy()
        i = 0

        while i < len(labels):
            label = labels[i]
            if label.startswith('B-'):
                entity_type = label[2:]
                # Find entity span
                j = i + 1
                while j < len(labels) and labels[j] == f'I-{entity_type}':
                    j += 1

                # Replace entity if pool exists
                if entity_type in entity_pool and entity_pool[entity_type]:
                    replacement = random.choice(entity_pool[entity_type])
                    replacement_tokens = replacement.split()
                    new_tokens[i:j] = replacement_tokens
                    # Note: This simplification doesn't adjust labels perfectly

                i = j
            else:
                i += 1

        return new_tokens, labels
    

    def augment(
            self,
            tokens: List[str],
            labels: List[str],
            methods: List[str] = None
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Apply multiple augmentation methods
        """
        if methods is None:
            methods = ['synonym', 'deletion', 'swap']

        augmented = []

        for method in methods:
            if method == 'synonym':
                aug_tokens, aug_labels = self.synonym_replacement(tokens, labels)
            elif method == 'deletion':
                aug_tokens, aug_labels = self.random_deletion(tokens, labels)
            elif method == 'swap':
                aug_tokens, aug_labels = self.random_swap(tokens, labels)
            else:
                continue

            augmented.append((aug_tokens, aug_labels))

        return augmented