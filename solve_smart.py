import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import override

Pos = tuple[int, int]


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


@dataclass
class Sequence:
    letter: str
    coords: list[Pos]
    rule: Rule

    def candidates(self, fixed: dict[Pos, int]):
        values = [fixed.get(p) for p in self.coords]
        template = ''.join(str(fixed.get(p, '{}')) for p in self.coords)
        results = []

        for p in product(range(10), repeat=values.count(None)):
            n = int(template.format(*map(str, p)))
            if self.rule.valid(n) and len(str(n)) == len(self.coords):
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
    return Sequence(letter, coords, RULES[letter])


def resolve_intersections(fixed: dict[Pos, int]):
    for p in sorted(set(G) - set(fixed)):
        s0, s1 = SEQS_AT[p]
        i0 = s0.coords.index(p)
        i1 = s1.coords.index(p)
        c0 = {str(c)[i0] for c in s0.candidates(fixed)}
        c1 = {str(c)[i1] for c in s1.candidates(fixed)}
        cands = c0 & c1

        if len(cands) == 1:
            v = list(cands)[0]
            print(f'Fixing {p} => {v}')
            fixed[p] = int(v)


def part2():
    init_correct_positions = {p for s in CORRECT for p in s.coords}
    fixed = {p: GUESSES[p] for p in G if p in init_correct_positions}

    while True:
        count = len(fixed)
        print('resolving...')
        resolve_intersections(fixed)
        if len(fixed) == count:
            break

    unfixed = set(G) - set(fixed)
    assert not unfixed, f'Grid is not fully constrained (missing {len(unfixed)})'

    out = ''.join(
        ''.join(str(fixed[r, c]) for c in range(COLS)) + '\n'
        for r in range(ROWS)
    )

    print(out)
    out = re.sub(r'[0,2,4,6,8]', '.', out)
    print(out)

    return sum(map(int, re.findall(r'\d+', out)))


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

# Construct letter => Rule mappings
RULES = {r.letter: r for r in map(Rule, rules_txt.split('\n'))}

# Collect vert/horiz letter sequences
SEQUENCES = {}
for letter, start in LETTER_STARTS.items():
    SEQUENCES[v.letter] = (v := extract_sequence(start, True))
    SEQUENCES[h.letter] = (h := extract_sequence(start, False))

# find the two sequences associated with each position
SEQS_AT = defaultdict(list)
for seq in SEQUENCES.values():
    for p in seq.coords:
        SEQS_AT[p].append(seq)

# Identify which sequences are correct/incorrect
CORRECT: list[Sequence] = []
INCORRECT: list[Sequence] = []

a1 = 0
for letter, rule in RULES.items():
    seq = SEQUENCES[letter]
    value = int(''.join(str(GUESSES[c]) for c in seq.coords))
    invalid = not rule.valid(value)
    a1 += value * invalid
    (CORRECT, INCORRECT)[invalid].append(seq)

print('part1:', a1)
print('part2:', a2 := part2())

assert a1 == 1401106
assert a2 == 517533251
