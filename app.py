pip install flask
from flask import Flask, render_template
from your_original_code import Scoring, Cipher
import string
import time
import math
import random
from collections import Counter
from typing import Dict, List, Tuple
from prettytable import PrettyTable
import numpy as np
import sympy
from scipy.stats import entropy as scipy_entropy
from itertools import combinations

app = Flask(__name__)

class Scoring:
    def __init__(self, cipher: 'Cipher', running_time: float, weights: Dict[str, float], max_length_multiplier: float = 2.0) -> None:
        self.cipher = cipher
        self.weights = weights
        self.max_length_multiplier = max_length_multiplier
        self.running_time = running_time
        self.frequency = Counter(self.cipher.encrypted_string)
        self.score = self.calculate_score()
        self.summary = self.generate_summary()

    def unique_chars_metric(self) -> float:
        return len(set(self.cipher.encrypted_string)) / len(set(string.printable))

    def distinct_sequences_metric(self) -> float:
        sequences = [''.join(seq) for seq in zip(self.cipher.encrypted_string, self.cipher.encrypted_string[1:])]
        return len(set(sequences)) / max(1, len(set(combinations(string.printable, 2))))

    def entropy_metric(self) -> float:
        probabilities = [count / len(self.cipher.encrypted_string) for count in self.frequency.values()]
        return scipy_entropy(probabilities, base=2) / math.log2(len(set(string.printable)))

    def frequency_analysis_metric(self) -> float:
        most_common_char_frequency = self.frequency.most_common(1)[0][1] / len(self.cipher.encrypted_string)
        return 1 - most_common_char_frequency

    def length_consistency_metric(self) -> float:
        return abs(len(self.cipher.encrypted_string) - len(self.cipher.original_string)) / len(self.cipher.original_string)

    def evenness_metric(self) -> float:
        frequencies = list(self.frequency.values())
        mean_freq = sum(frequencies) / len(frequencies)
        variance = sum((freq - mean_freq) ** 2 for freq in frequencies) / len(frequencies)
        return 1 - (math.sqrt(variance) / len(self.cipher.encrypted_string))

    def reversibility_metric(self) -> float:
        decrypted_string = self.cipher.decrypt()
        return float(decrypted_string == self.cipher.original_string)

    def change_propagation_metric(self) -> float:
        changed_string = 'a' + self.cipher.original_string[1:]
        cipher_changed = Cipher(changed_string)
        cipher_changed.encrypt()
        changed_encrypted_string = cipher_changed.encrypted_string
        levenshtein_distance = self._levenshtein_distance(self.cipher.encrypted_string, changed_encrypted_string)
        return levenshtein_distance / len(self.cipher.encrypted_string)

    def pattern_analysis_metric(self) -> float:
        pattern_instances_orig = sum(1 for a, b in zip(self.cipher.original_string, self.cipher.original_string[1:]) if a == b)
        pattern_instances_enc = sum(1 for a, b in zip(self.cipher.encrypted_string, self.cipher.encrypted_string[1:]) if a == b)

        if pattern_instances_orig == 0:
            return 1.0  # No repeated patterns in the original string
        elif pattern_instances_enc == 0:
            return 1.0  # No repeated patterns in the encrypted string
        else:
            return 1 - (pattern_instances_enc / pattern_instances_orig)


    def correlation_analysis_metric(self) -> float:
        correlations = sum(1 for a, b in zip(self.cipher.original_string, self.cipher.encrypted_string) if a == b)
        correlation_metric = correlations / len(self.cipher.original_string)
        return 1 - correlation_metric

    def complexity_metric(self) -> float:
        return len(self.cipher.encrypted_string) / len(self.cipher.original_string)

    def randomness_metric(self) -> float:
        random_string = ''.join(random.choice(string.printable) for _ in range(len(self.cipher.original_string)))
        cipher = Cipher(random_string)
        cipher.encrypt()
        random_encrypted_string = cipher.encrypted_string
        random_encrypted_frequency = Counter(random_encrypted_string)
        frequencies = list(random_encrypted_frequency.values())
        mean_freq = sum(frequencies) / len(frequencies)
        variance = sum((freq - mean_freq) ** 2 for freq in frequencies) / len(frequencies)
        return math.sqrt(variance) / len(string.printable)

    def normalized_levenshtein_metric(self) -> float:
        decrypted_string = self.cipher.decrypt()
        distance = self._levenshtein_distance(self.cipher.original_string, decrypted_string)
        return 1 - distance / max(len(self.cipher.original_string), len(decrypted_string))

    def encryption_consistency_metric(self) -> float:
        original_string = self.cipher.original_string
        original_encrypted_string = self.cipher.encrypted_string
        self.cipher.original_string = original_string
        self.cipher.encrypted_string = ""
        self.cipher.encrypt()
        second_encryption = self.cipher.encrypted_string
        self.cipher.encrypted_string = original_encrypted_string

        return float(original_encrypted_string == second_encryption)

    def running_time_metric(self) -> float:
        return np.clip(1 - self.running_time, 0, 1)


    def calculate_score(self) -> float:
        total_score = 0
        try:
            if len(self.cipher.encrypted_string) > len(self.cipher.original_string) * self.max_length_multiplier:
                raise Exception("The length of the encrypted string exceeds the allowed limit. This entry is disqualified.")

            total_score += self.weights["unique_chars"] * self.unique_chars_metric()
            total_score += self.weights["distinct_sequences"] * self.distinct_sequences_metric()
            total_score += self.weights["entropy"] * self.entropy_metric()
            total_score += self.weights["frequency_analysis"] * self.frequency_analysis_metric()
            total_score += self.weights["length_consistency"] * self.length_consistency_metric()
            total_score += self.weights["evenness"] * self.evenness_metric()
            total_score += self.weights["reversibility"] * self.reversibility_metric()
            total_score += self.weights["change_propagation"] * self.change_propagation_metric()
            total_score += self.weights["pattern_analysis"] * self.pattern_analysis_metric()
            total_score += self.weights["correlation_analysis"] * self.correlation_analysis_metric()
            total_score += self.weights["complexity"] * self.complexity_metric()
            total_score += self.weights["randomness"] * self.randomness_metric()
            total_score += self.weights["normalized_levenshtein"] * self.normalized_levenshtein_metric()
            total_score += self.weights["encryption_consistency"] * self.encryption_consistency_metric()
            total_score += self.weights["running_time"] * self.running_time_metric()
        except Exception as e:
            print(f"An error occurred while calculating the score: {e}")
            return 0

        return total_score

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


    def generate_summary(self) -> Dict[str, float]:
        summary = {
            "unique_chars": self.unique_chars_metric(),
            "distinct_sequences": self.distinct_sequences_metric(),
            "entropy": self.entropy_metric(),
            "frequency_analysis": self.frequency_analysis_metric(),
            "length_consistency": self.length_consistency_metric(),
            "evenness": self.evenness_metric(),
            "reversibility": self.reversibility_metric(),
            "change_propagation": self.change_propagation_metric(),
            "pattern_analysis": self.pattern_analysis_metric(),
            "correlation_analysis": self.correlation_analysis_metric(),
            "complexity": self.complexity_metric(),
            "randomness": self.randomness_metric(),
            "normalized_levenshtein": self.normalized_levenshtein_metric(),
            "encryption_consistency": self.encryption_consistency_metric(),
            "running_time": self.running_time_metric()
        }
        return summary


    @staticmethod
    def print_rsa_results(table_data: List[List[str]], scores: List[bool]) -> None:
        try:
            table = PrettyTable()
            table.field_names = ["Test No.", "Original String", "Encrypted String",
                                 "Decrypted String", "Decryption Success", "Running Time"]

            for data in table_data:
                table.add_row(data)

            print(table)
            print(f"\nAverage score: {sum(scores) / len(scores)}")
        except Exception as e:
            print(f"An error occurred while printing the results: {e}")

class Cipher:
    def __init__(self, original_string: str) -> None:
        self.original_string = original_string
        self.encrypted_string = ""

    def generate_keys(self):
        p = sympy.randprime(2**64, 2**128)
        q = sympy.randprime(2**64, 2**128)
        n = p * q
        phi = (p - 1) * (q - 1)

        e = sympy.randprime(3, phi - 1)
        d = sympy.mod_inverse(e, phi)

        return (e, n), (d, n)

    def encrypt(self, public_key):
        e, n = public_key
        encrypted_data = [pow(ord(char), e, n) for char in self.original_string]
        self.encrypted_string = ' '.join(map(str, encrypted_data))

    def decrypt(self, private_key):
        d, n = private_key
        encrypted_data = self.encrypted_string.split()
        decrypted_data = [chr(pow(int(char), d, n)) for char in encrypted_data]
        return ''.join(decrypted_data)

@app.route('/')
def index():
    max_length_multiplier = 2.0  # You can set the desired multiplier here
    sample_strings = [
        "Hello, World!",
        "This is a sample string",
        "Another string for testing",
        "A very very long string that should result in a high score",
        "Short string",
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "1234567890",
        "A string with special characters: !@#$%^&*()",
        "A string with spaces    between     words",
        "A string with a mix of letters, numbers, and special characters: abc123!@#",
    ]
    # Define the weights dictionary for scoring
    weights = {
        "unique_chars": 0.1,
        "distinct_sequences": 1.0,
        "entropy": 1.5,
        "frequency_analysis": 0.1,
        "length_consistency": 0.5,
        "evenness": 1.2,
        "reversibility": 2.0,
        "change_propagation": 1.5,
        "pattern_analysis": 1.5,
        "correlation_analysis": 1.5,
        "complexity": 1.0,
        "randomness": 1.5,
        "normalized_levenshtein": 1.0,
        "encryption_consistency": 2.0,
        "running_time": 0.5
    }

    rsa_scores = []
    rsa_table_data = []

    for i, original_string in enumerate(sample_strings, start=1):
        cipher = Cipher(original_string)
        public_key, private_key = cipher.generate_keys()
        cipher.encrypt(public_key)
        decrypted_string = cipher.decrypt(private_key)
        decryption_success = decrypted_string == original_string
        rsa_scores.append(decryption_success)
        rsa_table_data.append([
            i, original_string, cipher.encrypted_string,
            decrypted_string, decryption_success
        ])

    avg_score = sum(rsa_scores) / len(rsa_scores)
    return render_template('index.html', table_data=rsa_table_data, avg_score=avg_score)

if __name__ == "__main__":
    app.run(debug=True)
