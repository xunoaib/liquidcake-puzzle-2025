import math
import re
import sys
from dataclasses import dataclass
from itertools import count
from typing import override

from joblib import Memory
from z3 import And, Int, Or, Solver, Sum, sat

memory = Memory('.joblib')


def add(a: tuple[int, int], b: tuple[int, int]):
    return (a[0] + b[0]) % ROWS, (a[1] + b[1]) % COLS


@dataclass
class Sequence:
    letter: str
    coords: tuple[tuple[int, int], ...]


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


@memory.cache
def is_power(x: int, b: int):
    return any(b**e == x for e in range(x + 1))


def is_palindrome(x: int):
    return all(a == b for a, b in zip(str(x), str(x)[::-1]))


def extract_sequence(start: tuple[int, int], vert: bool):
    offset = (1, 0) if vert else (0, 1)
    coords = (start, )

    p = add(start, offset)
    while p not in LETTER_STARTS.values():
        coords += (p, )
        p = add(p, offset)

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

    ztemps = []  # temporary variables

    # for letter, rule in rules.items():
    #     if letter not in incorrect_letters:
    #         continue

    for letter in incorrect_letters:
        rule = rules[letter]
        seq = sequences[letter]

        z = Sum(zgrid[p] * 10**i for i, p in enumerate(seq.coords[::-1]))
        solver.add(z >= 10**(len(seq.coords) - 1))
        solver.add(z <= 10**(len(seq.coords) + 1))

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
                conds = []
                for e in count():
                    v = rule.of**e
                    if math.log10(v) > len(seq.coords):
                        break
                    conds.append(z == v)
                solver.add(Or(conds))
            case 'palindrome':
                solver.add(
                    And(
                        zgrid[p] == zgrid[q]
                        for p, q in list(zip(seq.coords, seq.coords[::-1]))
                        [:len(seq.coords) // 2]
                    )
                )
            case _:
                raise NotImplementedError(f'Unknown rule: {rule.type}')

    print('solving')
    assert solver.check() == sat, f'Model not satisfied: {solver.check()}'
    m = solver.model()

    seq = ''
    for r in range(ROWS):
        for c in range(COLS):
            v = zgrid[r, c]
            seq += str(v if isinstance(v, int) else m.evaluate(v))
        seq += '\n'

    print(seq)
    seq = re.sub(r'[0,2,4,6,8]', '.', seq)
    print(seq)
    return sum(map(int, re.findall(r'\d+', seq)))


if len(sys.argv) > 1:
    print('reading', sys.argv[1])
    file = open(sys.argv[1])
else:
    print('reading from stdin...')
    file = sys.stdin

a, b, guesses = file.read().strip().split('\n\n')

G = {
    (r, c): v
    for r, line in enumerate(a.split('\n'))
    for c, v in enumerate(line)
}

G_GUESSES = {
    (r, c): int(v)
    for r, line in enumerate(guesses.split('\n'))
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
rules: dict[str, Rule] = {r.letter: r for r in map(Rule, b.split('\n'))}

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
