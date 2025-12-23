import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from math import ceil, floor, log10
from typing import override

from z3 import Int, Or, Solver, Sum, sat

Pos = tuple[int, int]


@dataclass
class Sequence:
    letter: str
    coords: list[Pos]


class Rule:

    def __init__(self, text):
        m = re.match(
            r'(.*?) is a (square|cube|power|multiple|palindrome).*?(\d+)?$',
            text
        )
        assert m
        letter, _type, of = m.groups()
        self.letter = letter
        self.type = _type
        self._of = int(of) if of else None
        self.valid = {
            'square': lambda x: int(x**(1 / 2))**2 == x,
            'cube': lambda x: round(x**(1 / 3))**3 == x,
            'power': lambda x: is_power(x, b=self.of),
            'multiple': lambda x: x % self.of == 0,
            'palindrome': lambda x: str(x) == str(x)[::-1],
        }[_type]

    @property
    def of(self):
        assert self._of is not None
        return self._of

    @override
    def __repr__(self) -> str:
        if self._of is None:
            return f'<{self.letter} is {self.type}>'
        else:
            return f'<{self.letter} is {self.type} of {self.of}>'


def candidates(sequence: Sequence, rule: Rule, fixed: dict[Pos, int]):

    values = [fixed.get(p) for p in sequence.coords]
    nfree = values.count(None)
    template = ''.join(str(fixed.get(p, '{}')) for p in sequence.coords)
    results = []

    for p in product(range(10), repeat=nfree):
        n = int(template.format(*map(str, p)))
        if rule.valid(n):
            results.append(n)
    return results


def is_power(x: int, b: int):
    if x < 1 or b < 2:
        return False

    while x % b == 0:
        x //= b
    return x == 1


def add(a: Pos, b: Pos):
    return (a[0] + b[0]) % ROWS, (a[1] + b[1]) % COLS


def extract_sequence(start: Pos, vert: bool):
    offset = (1, 0) if vert else (0, 1)

    coords = [start]
    while (p := add(coords[-1], offset)) not in LETTER_STARTS.values():
        coords.append(p)

    letter = G[start].upper() if vert else G[start].lower()
    return Sequence(letter, coords)


def part2_smart():

    correct_positions = {p for s in correct for p in s.coords}
    fixed = {p: GUESSES[p] for p in G if p in correct_positions}

    for letter in sorted(sequences):
        cands = candidates(sequences[letter], rules[letter], fixed)
        print(letter, len(cands))

    exit()


if len(sys.argv) > 1:
    print('reading from', sys.argv[1])
    file = open(sys.argv[1])
else:
    print('reading from stdin...')
    file = sys.stdin

grid_txt, rules_txt, vals_txt = file.read().strip().split('\n\n')

G = {
    (r, c): v
    for r, line in enumerate(grid_txt.split('\n'))
    for c, v in enumerate(line)
}

GUESSES = {
    (r, c): int(v)
    for r, line in enumerate(vals_txt.split('\n'))
    for c, v in enumerate(line)
}

ROWS = 1 + max(r for r, _ in G)
COLS = 1 + max(c for _, c in G)

LETTER_STARTS = {v: k for k, v in G.items() if v != '.'}

# Collect vert/horiz letter sequences
sequences = {}
for letter, start in LETTER_STARTS.items():
    sequences[v.letter] = (v := extract_sequence(start, True))
    sequences[h.letter] = (h := extract_sequence(start, False))

# Construct letter => Rule mappings
rules = {r.letter: r for r in map(Rule, rules_txt.split('\n'))}

# Identify which sequences are correct/incorrect
correct: list[Sequence] = []
incorrect: list[Sequence] = []

a1 = 0
for letter, rule in rules.items():
    seq = sequences[letter]
    value = int(''.join(str(GUESSES[c]) for c in seq.coords))
    invalid = not rule.valid(value)
    a1 += value * invalid
    (correct, incorrect)[invalid].append(seq)

seqs_at = defaultdict(list)
for seq in sequences.values():
    for p in seq.coords:
        seqs_at[p].append(seq)

print('part1:', a1)
print('part2:', a2 := part2_smart())

assert a1 == 1401106
assert a2 == 517533251
