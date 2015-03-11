'''
'''
import codecs
import encodings
import re
import sys
import tokenize
import unittest
from collections import namedtuple

try:
    from cStringIO import StringIO
except:
    # py3
    from io import StringIO


__all__ = ('contract', )

KEYWORDS = ('require', 'ensure', 'body')
ENDMARKER = (tokenize.ENDMARKER, '')
ROWCOL_START = 2
ROWCOL_END = 3
LINE = 4

ENCODING_REGEX = re.compile('contract-([a-z0-9]+)')

Token = namedtuple('Token', ('type', 'string', 'tokenstart', 'tokenend', 'line'))
TokenPos = namedtuple('TokenPos', ('name_pos', 'block_indent', 'block_dedent'))


def contract_from_string(code):
    '''Simple wrapper for contract()'''
    return contract(StringIO(code).readline)


def keyword(name):
    return (1, name, (1, 0), (1, len(name)), name)


def op(name):
    return (51, name, (1, 0), (1, len(name)), name)


# To simplify token stream handling, all Operators and Delimiters tokens are returned using the generic token.OP token type.
CLOSE_PARENTESIS = op(')')
COLON = op(':')
COMMA = op(',')
DEF = keyword('def')
EQUAL = op('=')
NEWLINE = (4, '\n', (1, 0), (1, 1), '\n')
OPEN_PARENTESIS = op('(')
RETURN = keyword('return')


def contract(code):
    tokens_peek = []
    tokens_buffer = []
    decorators = []

    readline = (line.rstrip('\n \t').expandtabs() + "\n" for line in code).next
    token_generator = tokenize.generate_tokens(readline)

    def peek(iterations):
        missing = (iterations + 1) - len(tokens_peek)

        if missing > 0:
            for _ in range(missing):
                try:
                    tokens_peek.append(next(token_generator))
                except StopIteration:
                    return None

        if iterations < len(tokens_peek):
            return tokens_peek[iterations]

        return None

    def peek_name(iterations):
        value = peek(iterations)

        if value:
            return value[1]

    def is_token(position, type_, name):
        token_type, token_name, _, _, _ = peek(position)

        if token_type == type_ and token_name == name:
            return True
        return False

    def is_type(position, type_):
        token_type, _, _, _, _ = peek(position)

        if token_type == type_:
            return True
        return False

    def closing(position, open_, close):
        missing = 0

        while True:
            if open_(position):
                missing += 1

            if close(position):
                missing -= 1

            if missing:
                position += 1
            else:
                break

        return position

    def tokens_for(code):
        return list(tokenize.generate_tokens(StringIO(code).readline))

    def reformat(tokens, old_indent, new_indent):
        # (token type, token string, (tokenstart row, tokenstart col), (tokenend row, tokenend col), line)
        return (
            (type, string, srow_scol, erow_ecol, line.replace(old_indent, new_indent))
            for type, string, srow_scol, trow_tcol in tokens
        )

    def create_contract(name, arguments_tokens, require, ensure):
        # Details:
        #   - tokenize.generate_tokens() return ENDMARKER that we need to remove
        #   - we are using the INDET/DEDENT pair from the original source code (it is included in the boundaries)
        first = require or ensure
        last = ensure or require
        indent = tokens_peek[first[1]]
        dedent = tokens_peek[last[2]]

        name = '_{}_contract{}'.format(name, len(decorators))

        # def <decorator>(function):\n
        tokens = [
            DEF,
            keyword(name),
            OPEN_PARENTESIS,
            keyword('_function'),
            CLOSE_PARENTESIS,
            COLON,
            NEWLINE,
            indent,
        ]

        # def wrap(<args>):\n
        tokens.extend([
            DEF,
            keyword('wrap'),
            OPEN_PARENTESIS,
        ] + arguments_tokens + [
            CLOSE_PARENTESIS,
            COLON,
            NEWLINE,
            indent
        ])

        if require:
            # the slice is non-inclusive, the [2] is the actual index of the
            # dedent, so with this slice we do *not* include the dedent
            tokens.extend(tokens_peek[require[1]+1:require[2]])

        tokens.extend([
            keyword('result'),
            EQUAL,
            keyword('_function'),
            OPEN_PARENTESIS,
        ] + arguments_tokens + [
            CLOSE_PARENTESIS,
            NEWLINE,
        ])

        if ensure:
            tokens.extend(tokens_peek[ensure[1]+1:ensure[2]])

        tokens.extend([
            RETURN,
            keyword('result'),
            NEWLINE,
            dedent,

            RETURN,
            keyword('wrap'),
            NEWLINE,
            dedent,
        ])

        decorators.extend(tokens)

        return name

    open_parentheses = lambda pos: is_token(pos, tokenize.OP, '(')
    close_parentheses = lambda pos: is_token(pos, tokenize.OP, ')')
    open_ident = lambda pos: is_type(pos, tokenize.INDENT)
    close_ident = lambda pos: is_type(pos, tokenize.DEDENT)

    # 0   1   2
    # def name(...):
    #   body:
    #
    # After the colon we have a NEWLINE and then a INDENT

    while peek(0):
        if is_token(0, tokenize.NAME, 'def'):
            name = peek_name(1)
            colon_pos = closing(2, open_parentheses, close_parentheses) + 1
            arguments_tokens = tokens_peek[3:colon_pos-1]

            if is_token(colon_pos, tokenize.OP, ':'):
                function_start = colon_pos + 1 + 1  # + NEWLINE + INDENT
                maybe_next_pos = function_start + 1 # + NAME
                block_positions = {}
                tokens = []
                indent = peek(colon_pos + 1 + 1)

                while peek_name(maybe_next_pos) in KEYWORDS and is_token(maybe_next_pos + 1, tokenize.OP, ':'):
                    # we have a new block
                    name_pos = maybe_next_pos

                    if block_positions.get(peek_name(name_pos)) is not None:
                        raise SyntaxError('{} block defined more than once for the function {}'.format(peek_name(name_pos), name))

                    block_start = name_pos + 1 + 1 + 1 # colon + NEWLINE + INDENT
                    block_end = closing(block_start, open_ident, close_ident)
                    block_positions[peek_name(name_pos)] = TokenPos(name_pos, block_start, block_end)

                    maybe_next_pos = block_end + 1  # NAME

                if len(block_positions):
                    if not is_type(maybe_next_pos, tokenize.DEDENT):
                        raise SyntaxError('The function {} has code outside one of the blocks {}'.format(name, ', '.join(KEYWORDS)))

                    if 'body' not in block_positions:
                        raise SyntaxError('Missing body block for the function {}'.format(name))

                    if 'require' in block_positions or 'ensure' in block_positions:
                        decorator = create_contract(name, arguments_tokens, block_positions.get('require'), block_positions.get('ensure'))

                        tokens.append((tokenize.OP, '@', None, None, None))
                        tokens.append((tokenize.NAME, decorator, None, None, None))
                        tokens.append((tokenize.NL, '\n', None, None, None))

                    tokens.extend(tokens_peek[:function_start])

                    # using INDENT/DEDENT from the body block
                    body = block_positions['body']
                    tokens.extend(tokens_peek[body[1]:body[2]+1])

                    # replace old token with the new ones
                    tokens_peek = tokens

        tokens_buffer.extend(tokens_peek)
        tokens_peek = []

    for type_, name in reindent(decorators):
        yield type_, name

    for type_, name in reindent(tokens_buffer):
        yield type_, name


def contract_decoder(codec_decode):
    def decode(data):
        decoded, __ = codec_decode(data)
        return tokenize.untokenize(contract(decoded))
    return decode


def reindent(tokens):
    level = 0

    for type_, name, _, _, _ in tokens:
        if type_ == 5:
            level += 1
            yield type_, '    ' * level
        else:
            yield type_, name



def contract_codec(name, codec):
    '''Wrapper for a given encoder that will add the contract pre-processing'''
    decoder = contract_decoder(codec.decode)
    incrementaldecoder = type('IncrementalDecoder', (codecs.BufferedIncrementalDecoder,), {'decode': decoder})
    streamreader = type('StreamReader', (codecs.StreamReader, incrementaldecoder), {'decode': decoder})

    return codecs.CodecInfo(
        name=name,
        decode=decoder,
        incrementaldecoder=incrementaldecoder,
        streamreader=streamreader,

        # these need no changes
        encode=codec.encode,
        incrementalencoder=codec.incrementalencoder,
        streamwriter=codec.streamwriter,
    )


# This function is not exposed (will be None)
@codecs.register
def contract_search(codec):
    match = ENCODING_REGEX.match(codec)

    if match:
        encoding = encodings.search_function(match.group(1))
        return contract_codec(codec, encoding)


class ContractTestCase(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(
            list(contract('')),
            [ENDMARKER],
        )

    def test_one(self):
        self.assertEqual(
            list(contract('1')),
            [(tokenize.NUMBER, '1'), ENDMARKER],
        )

    def test_normal_function(self):
        code = 'def a(): pass'

        tokens = list(contract(code))
        self.assertEqual(
            tokens,
            [
                (tokenize.NAME, 'def'),
                (tokenize.NAME, 'a'),
                (tokenize.OP, '('),
                (tokenize.OP, ')'),
                (tokenize.OP, ':'),
                (tokenize.NAME, 'pass'),
                ENDMARKER,
            ]
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False, help='flag to run the tests')
    parser.add_argument('--failfast', action='store_true', default=False, help='unittest failfast')
    parser.add_argument('file', help='File to be preprocessed', nargs='?')
    args = parser.parse_args()

    if args.test:
        import doctest
        (failures, total) = doctest.testmod()

        if failures:
            sys.exit(failures)

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractTestCase)
        result = unittest.TextTestRunner(failfast=args.failfast).run(suite)

        if result.errors or result.failures:
            sys.exit(len(result.errors) + len(result.failures))

    elif args.file:
        with open(args.file) as handler:
            print(tokenize.untokenize(contract(handler)))

    else:
        parser.print_help()
