import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import override

Pos = tuple[int, int]


class Rule:

    RULES = {
        'square': lambda x, _: int(x**(1 / 2))**2 == x,
        'cube': lambda x, _: round(x**(1 / 3))**3 == x,
        'power': lambda x, b: is_power(x, b),
        'multiple': lambda x, b: x % b == 0,
        'palindrome': lambda x, _: str(x) == str(x)[::-1],
    }

    def __init__(self, text):
        m = re.match(
            r'(.*?) is a (square|cube|power|multiple|palindrome)(?: of (\d+))?$',
            text
        )
        assert m, text
        self.letter, self.type, of = m.groups()
        self.of = int(of) if of else None

    def valid(self, x: int):
        return self.RULES[self.type](x, self.of)

    @override
    def __repr__(self) -> str:
        of = f' of {self.of}' if self.of else ''
        return f'<{self.letter} is {self.type}{of}>'


@dataclass
class Sequence:
    letter: str
    coords: list[Pos]
    rule: Rule
    _candidates: list[str] | None = None
    index: dict[Pos, int] = field(default_factory=dict)

    def __post_init__(self):
        self.index = {p: i for i, p in enumerate(self.coords)}

    def all_candidates(self) -> list[str]:
        out = []
        for digits in product('0123456789', repeat=len(self.coords)):
            if digits[0] == '0':
                continue
            s = ''.join(digits)
            if self.rule.valid(int(s)):
                out.append(s)
        return out

    def filter_candidates(self, fixed: dict[Pos, str]):
        assert self._candidates
        vals = [fixed.get(p) for p in self.coords]
        return [
            c for c in self._candidates
            if all(v is None or v == c for v, c in zip(vals, c))
        ]

    def candidates(self, fixed: dict[Pos, str]):
        if self._candidates is None:
            self._candidates = self.all_candidates()
        self._candidates = self.filter_candidates(fixed)
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
    letter = STARTS[start].upper() if vert else STARTS[start].lower()
    offset = (1, 0) if vert else (0, 1)
    coords = [start]
    while (p := add(coords[-1], offset)) not in STARTS:
        coords.append(p)
    return Sequence(letter, coords, RULES[letter])


def resolve_intersections(fixed: dict[Pos, str]):
    changed = False
    for p in sorted(set(GUESSES) - set(fixed)):
        cands = cell_candidates(p, fixed)
        if len(cands) == 1:
            fixed[p] = (v := cands.pop())
            changed = True
            print(f'Fixing {p} => {v} ({len(GUESSES)-len(fixed)} left)')
    return changed


def cell_candidates(p: Pos, fixed: dict[Pos, str]):
    s0, s1 = SEQS_AT[p]
    i0 = s0.index[p]
    i1 = s1.index[p]
    c0 = {str(c)[i0] for c in s0.candidates(fixed)}
    c1 = {str(c)[i1] for c in s1.candidates(fixed)}
    return c0 & c1


def part2(fixed: dict[Pos, str]):

    print('Resolving...')
    while resolve_intersections(fixed):
        print('Resolving...')

    if (unfixed := set(GUESSES) - set(fixed)):
        print(f'\n\033[93mWarn: Some cells are not fully constrained\033[0m')
        print('Cells:', unfixed)
        for p in unfixed:
            cands = cell_candidates(p, fixed)
            print(f'cell candidates @ {p} = {cands}')
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

STARTS = {
    (r, c): v
    for r, line in enumerate(grid_txt.split('\n'))
    for c, v in enumerate(line) if v != '.'
}

GUESSES = {
    (r, c): v
    for r, line in enumerate(vals_txt.split('\n'))
    for c, v in enumerate(line)
}

ROWS = 1 + max(r for r, _ in GUESSES)
COLS = 1 + max(c for _, c in GUESSES)

RULES = {r.letter: r for r in map(Rule, rules_txt.split('\n'))}

# Collect vert/horiz letter sequences
SEQS_AT: dict[Pos, list[Sequence]] = defaultdict(list)
sequences = []
for pos, ch in STARTS.items():
    if ch != '.':
        for dir in [True, False]:
            seq = extract_sequence(pos, dir)
            sequences.append(seq)
            for p in seq.coords:
                SEQS_AT[p].append(seq)

fixed = {}
a1 = 0

for seq in sequences:
    value = int(''.join(GUESSES[p] for p in seq.coords))
    if seq.rule.valid(value):
        fixed.update({p: GUESSES[p] for p in seq.coords})
    else:
        a1 += value

print('part1:', a1)
print('part2:', a2 := part2(fixed))
# print('part2:', a2 := part2({}))  # also possible

assert a1 == 1401106
assert a2 == 517533251
