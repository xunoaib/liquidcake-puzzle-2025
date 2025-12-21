import re
import sys
from dataclasses import dataclass
from math import ceil, floor, log10
from typing import override

from z3 import And, Int, Or, Solver, Sum, sat


@dataclass
class Sequence:
    letter: str
    coords: list[tuple[int, int]]


class Rule:

    def __init__(self, text):
        m = re.match(
            r'(.*?) is a (square|cube|power|multiple|palindrome).*?(\d+)?$',
            text
        )
        assert m
        self.letter = m.group(1)
        self.type = m.group(2)
        self._of = None if m.group(3) is None else int(m.group(3))
        self.text = text
        self.valid = {
            'square': lambda x: int(x**(1 / 2))**2 == x,
            'cube': lambda x: int(x**(1 / 3))**3 == x,
            'power': lambda x: is_power(x, b=self.of),
            'multiple': lambda x: x % self.of == 0,
            'palindrome': is_palindrome,
        }[self.type]

    @property
    def of(self):
        assert self._of is not None
        return self._of

    @override
    def __repr__(self) -> str:
        if self._of is None:
            return f'<{self.letter} is {self.type})>'
        else:
            return f'<{self.letter} is {self.type} of {self.of}>'


def is_power(x: int, b: int):
    if x < 1 or b < 2:
        return False

    while x % b == 0:
        x //= b
    return x == 1


def is_palindrome(x: int):
    return str(x) == str(x)[::-1]


def add(a: tuple[int, int], b: tuple[int, int]):
    return (a[0] + b[0]) % ROWS, (a[1] + b[1]) % COLS


def extract_sequence(start: tuple[int, int], vert: bool):
    offset = (1, 0) if vert else (0, 1)
    coords = [start]
    while (p := add(coords[-1], offset)) not in LETTER_STARTS.values():
        coords.append(p)

    letter = G[start].upper() if vert else G[start].lower()
    return Sequence(letter, coords)


def part2():
    correct_positions = {p for s in correct for p in s.coords}
    incorrect_letters = {s.letter for s in incorrect}
    incorrect_positions = {
        p
        for s in incorrect
        for p in s.coords
    } - correct_positions

    solver = Solver()

    # Create a grid where correct positions are fixed ints
    # and incorrect positions are symbolic variables
    zgrid = {
        p:
        (G_GUESSES[p] if p in correct_positions else Int(f'p_{p[0]}_{p[1]}'))
        for p in G
    }

    for p in incorrect_positions:
        solver.add(zgrid[p] >= 0)
        solver.add(zgrid[p] <= 9)

    ztemps = []  # temporary z3 variables

    for letter in incorrect_letters:
        rule = rules[letter]
        seq = sequences[letter]

        z = Sum(zgrid[p] * 10**i for i, p in enumerate(seq.coords[::-1]))

        zdigits = len(seq.coords)
        solver.add(z >= 10**(zdigits - 1))
        solver.add(z <= 10**(zdigits + 1))

        match rule.type:
            case 'cube':
                ztemps.append(t := Int(f'tmp_{len(ztemps)}'))
                solver.add(z == t * t * t)
            case 'square':
                ztemps.append(t := Int(f'tmp_{len(ztemps)}'))
                solver.add(z == t * t)
            case 'multiple':
                solver.add(z % rule.of == 0)
            case 'power':
                b = rule.of
                lo = ceil((zdigits - 1) / log10(b))
                hi = floor(zdigits / log10(b))
                solver.add(Or(z == b**exp for exp in range(lo, hi + 1)))
            case 'palindrome':
                half = len(seq.coords) // 2
                solver.add(
                    And(
                        zgrid[p] == zgrid[q] for p, q in
                        zip(seq.coords[:half], seq.coords[:half:-1])
                    )
                )
            case _:
                raise NotImplementedError(f'Unknown rule: {rule.type}')

    print('solving')
    res = solver.check()
    assert res == sat, f'Model not satisfied: {res}'
    m = solver.model()

    seq = ''
    for r in range(ROWS):
        for c in range(COLS):
            v = zgrid[r, c]
            seq += str(v if isinstance(v, int) else m[v])
        seq += '\n'

    print(seq)
    seq = re.sub(r'[0,2,4,6,8]', '.', seq)
    print(seq)
    return sum(map(int, re.findall(r'\d+', seq)))


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

G_GUESSES = {
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
rules: dict[str, Rule] = {
    r.letter: r
    for r in map(Rule, rules_txt.split('\n'))
}

# Identify which sequences are correct/incorrect
correct: list[Sequence] = []
incorrect: list[Sequence] = []

a1 = 0
for letter, rule in rules.items():
    seq = sequences[letter]
    value = int(''.join(str(G_GUESSES[c]) for c in seq.coords))
    invalid = not rule.valid(value)
    a1 += value * invalid
    (correct, incorrect)[invalid].append(seq)

print('part1:', a1)
print('part2:', a2 := part2())

assert a1 == 1401106
assert a2 == 517533251
