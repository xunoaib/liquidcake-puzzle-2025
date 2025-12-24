import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import override

Pos = tuple[int, int]

RULE_PREDICATES = {
    'square': lambda x, _: int(x**(1 / 2))**2 == x,
    'cube': lambda x, _: round(x**(1 / 3))**3 == x,
    'power': lambda x, b: is_power(x, b),
    'multiple': lambda x, b: x % b == 0,
    'palindrome': lambda x, _: str(x) == str(x)[::-1],
}


class Rule:

    def __init__(self, text):
        m = re.match(
            r'(.*?) is a (square|cube|power|multiple|palindrome)(?: of (\d+))?$',
            text
        )
        assert m, text
        self.letter, self.type, of = m.groups()
        self.of = int(of) if of else None

    def valid(self, x: int):
        return RULE_PREDICATES[self.type](x, self.of)

    @override
    def __repr__(self) -> str:
        of = f' of {self.of}' if self.of else ''
        return f'<{self.letter} is {self.type}{of}>'


@dataclass
class Sequence:
    letter: str
    coords: list[Pos]
    rule: Rule
    index: dict[Pos, int] = field(init=False)
    _all: list[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.index = {p: i for i, p in enumerate(self.coords)}

    def all_candidates(self) -> list[str]:
        if not self._all:
            if self.rule.type == 'square':
                return list(exponent_candidates(len(self.coords), 2))
            if self.rule.type == 'cube':
                return list(exponent_candidates(len(self.coords), 3))

            for digits in product('0123456789', repeat=len(self.coords)):
                if digits[0] == '0':
                    continue
                s = ''.join(digits)
                if self.rule.valid(int(s)):
                    self._all.append(s)
        return self._all

    def filter_candidates(self, known: dict[Pos, str]):
        assert self._all
        vals = [known.get(p) for p in self.coords]
        return [
            c for c in self._all
            if all(v is None or v == c for v, c in zip(vals, c))
        ]

    def candidates(self, known: dict[Pos, str]):
        if not self._all:
            self._all = self.all_candidates()
        self._all = self.filter_candidates(known)
        return self._all


def exponent_candidates(ndigits, exponent):
    lo = round((10**(ndigits - 1))**(1 / exponent))
    hi = round((10**ndigits - 1)**(1 / exponent))
    for k in range(lo, hi + 1):
        v = k**exponent
        if len(str(v)) == ndigits:
            yield str(v)


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


def resolve_intersections(known: dict[Pos, str]):
    updated = False
    for p in sorted(GUESSES.keys() - known.keys()):
        cands = cell_candidates(p, known)
        if len(cands) == 1:
            known[p] = next(iter(cands))
            updated = True
            print(f'Fixing {p} => {known[p]} ({len(GUESSES)-len(known)} left)')
    return updated


def cell_candidates(p: Pos, known: dict[Pos, str]):
    s0, s1 = SEQS_AT[p]
    i0 = s0.index[p]
    i1 = s1.index[p]
    c0 = {str(c)[i0] for c in s0.candidates(known)}
    c1 = {str(c)[i1] for c in s1.candidates(known)}
    return c0 & c1


def part2(known: dict[Pos, str]):

    while resolve_intersections(known):
        pass

    if unknown := set(GUESSES) - set(known):
        print(f'\n\033[93mSome cells are not fully constrained\033[0m')
        print('Cells:', unknown)
        for p in unknown:
            cands = cell_candidates(p, known)
            print(f'cell candidates @ {p} = {cands}')
            known[p] = next(iter(cands))
            print(f'Arbitrarily fixing {p} => {known[p]}')

    out = grid_to_str(known)
    masked = re.sub(r'[0,2,4,6,8]', '.', out)

    print()
    print(out)
    print()
    print(masked)
    print()

    return sum(map(int, re.findall(r'\d+', masked)))


def grid_to_str(grid: dict[Pos, str]):
    return '\n'.join(
        ''.join(grid[r, c] for c in range(COLS)) for r in range(ROWS)
    )


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

a1 = 0
known = {}

# Collect vert/horiz letter sequences
SEQS_AT: dict[Pos, list[Sequence]] = defaultdict(list)
for pos, ch in STARTS.items():
    for dir in [True, False]:
        seq = extract_sequence(pos, dir)
        for p in seq.coords:
            SEQS_AT[p].append(seq)

        # validate rule for part 1
        value = int(''.join(GUESSES[p] for p in seq.coords))
        if seq.rule.valid(value):
            known.update({p: GUESSES[p] for p in seq.coords})
        else:
            a1 += value

print('part1:', a1)
print('part2:', a2 := part2(known))  # fix known values
# print('part2:', a2 := part2({}))  # also solvable w/o any values

assert a1 == 1401106
assert a2 == 517533251
