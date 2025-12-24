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
    _candidates: list[str] | None = None

    def all_candidates(self) -> list[str]:
        template = '{}' * len(self.coords)
        candidates = []
        for p in product(range(10), repeat=len(self.coords)):
            s = template.format(*map(str, p))
            n = int(s)
            if self.rule.valid(n) and len(str(n)) == len(self.coords):
                candidates.append(s)
        return candidates

    def new_candidates(self, fixed: dict[Pos, str]) -> list[str]:
        assert self._candidates
        vals = list(map(fixed.get, self.coords))
        return [
            n for n in self._candidates
            if all(v == c for v, c in zip(vals, n) if v is not None)
        ]

    def candidates(self, fixed: dict[Pos, str]):
        if self._candidates is None:
            self._candidates = self.all_candidates()

        self._candidates = self.new_candidates(fixed)
        return self._candidates


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


def resolve_intersections(fixed: dict[Pos, str]):
    for p in sorted(set(G) - set(fixed)):
        s0, s1 = SEQS_AT[p]
        i0 = s0.coords.index(p)
        i1 = s1.coords.index(p)
        c0 = {str(c)[i0] for c in s0.candidates(fixed)}
        c1 = {str(c)[i1] for c in s1.candidates(fixed)}
        cands = c0 & c1

        if len(cands) == 1:
            fixed[p] = (v := cands.pop())
            print(f'Fixing {p} => {v} ({len(G)-len(fixed)} left)')


def part2(fixed: dict[Pos, str]):
    while True:
        count = len(fixed)
        print('resolving...')
        resolve_intersections(fixed)
        if len(fixed) == count:
            break

    unfixed = set(G) - set(fixed)
    if unfixed:
        print(
            f'\n\033[93mWarn: Some cells are not fully constrained: {unfixed}\033[0m'
        )
        for p in unfixed:
            s0, s1 = SEQS_AT[p]
            i0 = s0.coords.index(p)
            i1 = s1.coords.index(p)
            c0 = {str(c)[i0] for c in s0.candidates(fixed)}
            c1 = {str(c)[i1] for c in s1.candidates(fixed)}
            cands = c0 & c1

            print()
            print('s0 =', s0)
            print('s1 =', s1)
            print()
            print(
                's0 candidates: str index ', s0.coords.index(p), 'of',
                s0.candidates(fixed)
            )
            print(
                's1 candidates: str index ', s1.coords.index(p), 'of',
                s1.candidates(fixed)
            )
            print()
            print('s0 cell candidates:', c0)
            print('s1 cell candidates:', c1)
            print('final cell candidates =', cands)

            fixed[p] = (v := cands.pop())
            print(f'Arbitrarily fixing {p} => {v}')

    out = ''.join(
        ''.join(str(fixed[r, c]) for c in range(COLS)) + '\n'
        for r in range(ROWS)
    )

    print()
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
    (r, c): v
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
SEQS_AT: dict[Pos, list[Sequence]] = defaultdict(list)
for seq in SEQUENCES.values():
    for p in seq.coords:
        SEQS_AT[p].append(seq)

# Identify which sequences are correct/incorrect
CORRECT_POSITIONS: set[Pos] = set()

a1 = 0
for letter, rule in RULES.items():
    seq = SEQUENCES[letter]
    value = int(''.join(str(GUESSES[c]) for c in seq.coords))
    invalid = not rule.valid(value)
    a1 += value * invalid
    if not invalid:
        CORRECT_POSITIONS.update({p for p in seq.coords})

fixed = {p: GUESSES[p] for p in G if p in CORRECT_POSITIONS}
# fixed = {}  # also possible

print('part1:', a1)
print('part2:', a2 := part2(fixed))

assert a1 == 1401106
assert a2 == 517533251
