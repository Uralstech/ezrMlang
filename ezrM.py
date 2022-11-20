from vendor.string_with_underline import str_w_undln
from os import path, system, name, getcwd
from string import ascii_letters
from importlib import util

# CONSTANTS

VERSION = '1.0.0.0.5'
VERSION_DATE = '20-11-22'
NUMBERS = '0123456789'
ALPHABETS = ''.join((chr(i) for i in range(3328, 3455))) + ascii_letters
ALPHANUM = ALPHABETS + NUMBERS
ALPHANUM_UNDERSCORE = ALPHANUM + '_'
NUMBERS_DOT = NUMBERS + '.'
LIB_PATH = path.join(path.dirname(path.abspath(__file__)), 'Libraries')

# ERRORS

class Error:
	def __init__(self, err_name, start_pos, end_pos, details):
		self.err_name = err_name
		self.start_pos = start_pos
		self.end_pos = end_pos
		self.details = details

	def as_string(self):
		return f'{self.err_name}: {self.details} (നിര {self.start_pos.fn} | വരി {self.start_pos.ln+1})\n\n{str_w_undln(self.start_pos.ftxt, self.start_pos, self.end_pos)}'

class UnkownCharError(Error):
	def __init__(self, start_pos, end_pos, details):
		super().__init__('അജ്ഞാത സ്വഭാവം', start_pos, end_pos, details)

class InvalidSyntaxError(Error):
	def __init__(self, start_pos, end_pos, details):
		super().__init__('അസാധുവായ വാക്യഘടന', start_pos, end_pos, details)
		
RTE_DEFAULT        = 'നിർവ്വഹണ-സമയം'
RTE_CUSTOM         = 'മറ്റുള്ളവ'
RTE_DICTKEY        = 'നിഘണ്ടു'
RTE_ILLEGALOP      = 'നിയമവിരുദ്ധമായ-പ്രവർത്തനം'
RTE_UNDEFINEDVAR   = 'നിർവചിക്കാത്ത-ഇനം'
RTE_IDXOUTOFRANGE  = 'പരിധിക്ക്-പുറത്തുള്ള-സൂചിക'
RTE_TOOMANYARGS    = 'വളരെയധികം-വാദങ്ങൾ'
RTE_TOOFEWARGS     = 'വളരെ-കുറച്ച്-വാദങ്ങൾ'
RTE_INCORRECTTYPE  = 'തെറ്റായ-തരം'
RTE_MATH		   = 'കണക്ക്'
RTE_IO	   		   = 'ഇടുക-എടുക്കുക'

class RuntimeError(Error):
	def __init__(self, start_pos, end_pos, error_type, details, context):
		super().__init__('നിർവ്വഹണ സമയ പിശക്', start_pos, end_pos, details)
		self.error_type = error_type
		self.context = context

	def as_string(self):
		return f'{self.generate_traceback()}{self.err_name}: {self.details}\n\n{str_w_undln(self.start_pos.ftxt, self.start_pos, self.end_pos)}'

	def generate_traceback(self):
		result = ''
		pos = self.start_pos
		context = self.context

		while context:
			result = f'\tനിര {pos.fn} | വരി {str(pos.ln+1)} | {context.display_name} ൽ\n{result}'
			pos = context.parent_entry_pos
			context = context.parent

		return f'ഉറവിടം തേട (ഏറ്റവും പുതിയ വിളി അവസാനമായി):\n{result}'

# POSITION

class Position:
	def __init__(self, idx, ln, col, fn, ftxt):
		self.idx = idx
		self.ln = ln
		self.col = col
		self.fn = fn
		self.ftxt = ftxt

	def advance(self, current_char=None):
		self.idx += 1
		self.col += 1

		if current_char == '\n':
			self.ln += 1
			self.col = 0

		return self

	def copy(self):
		return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

# TOKEN

TT_INT     = 'INT'
TT_FLOAT   = 'FLOAT'
TT_STRING  = 'STRING'
TT_ID      = 'ID'
TT_KEY     = 'KEY'
TT_PLUS    = 'PLUS'
TT_MINUS   = 'MINUS'
TT_MUL     = 'MUL'
TT_DIV     = 'DIV'
TT_MOD     = 'MOD'
TT_POW     = 'POW'
TT_COLON    = 'EQUALS'
TT_LPAREN  = 'LPAREN'
TT_RPAREN  = 'RPAREN'
TT_LSQUARE = 'LSQUARE'
TT_RSQUARE = 'RSQUARE'
TT_LCURLY  = 'LCURLY'
TT_RCURLY  = 'RCURLY'
TT_IE      = 'IE'
TT_NE      = 'NE'
TT_LT      = 'LT'
TT_GT      = 'GT'
TT_LTE     = 'LTE'
TT_GTE     = 'GTE'
TT_COMMA   = 'COMMA'
TT_DOT     = 'DOT'
TT_NEWLINE = 'NEWLINE'
TT_EOF     = 'EOF'

#          ['item',  'and','or',           'invert',     'if',       'else',   'do',       'count',    'from',  'as',       'to',   'step',   'while',    'function',     'with',   'end',         'return',       'skip',      'stop',     'try',        'error',   'in', 'object', 'global',  'include']
KEYWORDS = ['ഇനം', 'ഉം', 'അല്ലെങ്കിൽ', 'വിപരീതം', 'എങ്കിൽ', 'വേറെ', 'ചെയ്യുക', 'എണ്ണുക', 'നിന്ന്', 'പോലെ', 'വരെ', 'ഘട്ടം', 'എന്നാൽ', 'പ്രവർത്തനം', 'കൂടെ', 'അവസാനം', 'കൊടുക്കുക', 'തുടരുക', 'നിർത്തൂ', 'ശ്രമിക്കുക', 'പിശക്', 'ൽ', 'വസ്തു', 'ലോകം', 'ഉൾപ്പെടുന്നു']

class Token:
	def __init__(self, type_, value=None, start_pos=None, end_pos=None):
		self.type = type_
		self.value = value

		if start_pos:
			self.start_pos = start_pos.copy()
			self.end_pos = start_pos.copy()
			self.end_pos.advance()
		if end_pos: self.end_pos = end_pos

	def matches(self, type_, value):
		return self.type == type_ and self.value == value

	def __repr__(self):
		if self.value: return f'{self.type}:{self.value}'
		return f'{self.type}'
		
	def __hash__(self):
		return hash(self.type) ^ hash(self.value)

# LEXER

class Lexer:
	def __init__(self, fn, input_):
		self.fn = fn
		self.input = input_
		self.pos = Position(-1, 0, -1, fn, input_)
		self.current_char = None
		self.advance()

	def advance(self):
		self.pos.advance(self.current_char)
		self.current_char = self.input[self.pos.idx] if self.pos.idx < len(self.input) else None

	def compile_tokens(self):
		tokens = []

		while self.current_char != None:
			if self.current_char in ' \t':
				self.advance()
			elif self.current_char == '@':
				self.skip_comment()
			elif self.current_char in ';\n':
				tokens.append(Token(TT_NEWLINE, start_pos=self.pos))
				self.advance()
			elif self.current_char == '\'' or self.current_char == '"':
				tokens.append(self.compile_string())
			elif self.current_char == '+':
				tokens.append(Token(TT_PLUS, start_pos=self.pos))
				self.advance()
			elif self.current_char == '-':
				tokens.append(Token(TT_MINUS, start_pos=self.pos))
				self.advance()
			elif self.current_char == '*':
				tokens.append(Token(TT_MUL, start_pos=self.pos))
				self.advance()
			elif self.current_char == '/':
				tokens.append(Token(TT_DIV, start_pos=self.pos))
				self.advance()
			elif self.current_char == '%':
				tokens.append(Token(TT_MOD, start_pos=self.pos))
				self.advance()
			elif self.current_char == '^':
				tokens.append(Token(TT_POW, start_pos=self.pos))
				self.advance()
			elif self.current_char == ':':
				tokens.append(Token(TT_COLON, start_pos=self.pos))
				self.advance()
			elif self.current_char == '(':
				tokens.append(Token(TT_LPAREN, start_pos=self.pos))
				self.advance()
			elif self.current_char == ')':
				tokens.append(Token(TT_RPAREN, start_pos=self.pos))
				self.advance()
			elif self.current_char == '[':
				tokens.append(Token(TT_LSQUARE, start_pos=self.pos))
				self.advance()
			elif self.current_char == ']':
				tokens.append(Token(TT_RSQUARE, start_pos=self.pos))
				self.advance()
			elif self.current_char == '{':
				tokens.append(Token(TT_LCURLY, start_pos=self.pos))
				self.advance()
			elif self.current_char == '}':
				tokens.append(Token(TT_RCURLY, start_pos=self.pos))
				self.advance()
			elif self.current_char == '=':
				tokens.append(Token(TT_IE, start_pos=self.pos))
				self.advance()
			elif self.current_char == '!':
				tokens.append(Token(TT_NE, start_pos=self.pos))
				self.advance()
			elif self.current_char == ',':
				tokens.append(Token(TT_COMMA, start_pos=self.pos))
				self.advance()
			elif self.current_char == '.':
				tokens.append(Token(TT_DOT, start_pos=self.pos))
				self.advance()
			elif self.current_char == '<':
				tokens.append(self.compile_less_than())
			elif self.current_char == '>':
				tokens.append(self.compile_greater_than())
			elif self.current_char in NUMBERS:
				tokens.append(self.compile_num())
			elif self.current_char in ALPHABETS:
				tokens.append(self.compile_identifier())
			else:
				pos_start = self.pos.copy()
				char = self.current_char
				self.advance()
				return [], UnkownCharError(pos_start, self.pos, f"'{char}'")

		tokens.append(Token(TT_EOF, start_pos=self.pos))
		return tokens, None

	def compile_string(self):
		string_to_return = ''
		start_pos = self.pos.copy()
		escape_char = False
		string_char = self.current_char
		self.advance()

		escape_chars = {'n':'\n', 't':'\t', 'r': '\r'}
		while self.current_char != None and (self.current_char != string_char or escape_char):
			if escape_char:
				string_to_return += escape_chars.get(self.current_char, self.current_char)
				escape_char = False
			else:
				if self.current_char == '\\':
					escape_char = True
				else:
					string_to_return += self.current_char

			self.advance()
		
		self.advance()
		return Token(TT_STRING, string_to_return, start_pos, self.pos)

	def compile_less_than(self):
		start_pos = self.pos.copy()
		self.advance()

		if self.current_char == '=':
			self.advance()
			return Token(TT_LTE, start_pos=start_pos, end_pos=self.pos)
		
		return Token(TT_LT, start_pos=start_pos, end_pos=self.pos)

	def compile_greater_than(self):
		start_pos = self.pos.copy()
		self.advance()

		if self.current_char == '=':
			self.advance()
			return Token(TT_GTE, start_pos=start_pos, end_pos=self.pos)
		
		return Token(TT_GT, start_pos=start_pos, end_pos=self.pos)

	def compile_identifier(self):
		id_str = ''
		start_pos = self.pos.copy()

		while self.current_char != None and self.current_char in ALPHANUM_UNDERSCORE:
			id_str += self.current_char
			self.advance()

		token_type = TT_KEY if id_str in KEYWORDS else TT_ID
		return Token(token_type, id_str, start_pos, self.pos)

	def compile_num(self):
		num_str = ''
		dot_count = 0
		start_pos = self.pos.copy()

		while self.current_char != None and self.current_char in NUMBERS_DOT:
			if self.current_char == '.':
				if dot_count == 1: break
				dot_count += 1

			num_str += self.current_char
			self.advance()

		if dot_count == 0: return Token(TT_INT, int(num_str), start_pos, self.pos)
		return Token(TT_FLOAT, float(num_str), start_pos, self.pos)

	def skip_comment(self):
		self.advance()
		while self.current_char != '\n' and self.current_char != None: self.advance()

# NODES

class NumberNode:
	def __init__(self, token, start_pos, end_pos):
		self.token = token
		
		self.start_pos = start_pos
		self.end_pos = end_pos

	def __repr__(self):
		return f'{self.token}'

	def __hash__(self):
		return hash(self.token)

class StringNode:
	def __init__(self, token, start_pos, end_pos):
		self.token = token

		self.start_pos = start_pos
		self.end_pos = end_pos

	def __repr__(self):
		return f'{self.token}'
		
	def __hash__(self):
		return hash(self.token)

class ArrayNode:
	def __init__(self, element_nodes, start_pos, end_pos):
		self.element_nodes = element_nodes

		self.start_pos = start_pos
		self.end_pos = end_pos
	
	def __hash__(self):
		hash_value = hash(0)
		for i in self.element_nodes: hash_value ^= hash(i)

		return hash_value

class ListNode:
	def __init__(self, element_nodes, start_pos, end_pos):
		self.element_nodes = element_nodes

		self.start_pos = start_pos
		self.end_pos = end_pos
	
	def __hash__(self):
		hash_value = hash(0)
		for i in self.element_nodes: hash_value ^= hash(i)

		return hash_value

class DictNode:
	def __init__(self, pair_nodes, start_pos, end_pos):
		self.pair_nodes = pair_nodes

		self.start_pos = start_pos
		self.end_pos = end_pos
	
	def __hash__(self):
		hash_value = hash(0)
		for key, value in self.pair_nodes: hash_value ^= hash(key) ^ hash(value)

		return hash_value

class VarAccessNode:
	def __init__(self, var_name_token, start_pos, end_pos):
		self.var_name_token = var_name_token

		self.start_pos = start_pos
		self.end_pos = end_pos

	def __hash__(self):
		return hash(self.var_name_token)

class VarAssignNode:
	def __init__(self, var_name_token, value_node, is_global, start_pos, end_pos):
		self.var_name_token = var_name_token
		self.value_node = value_node
		self.is_global = is_global

		self.start_pos = start_pos
		self.end_pos = end_pos

	def __hash__(self):
		return hash(self.var_name_token) ^ hash(self.value_node) ^ hash(self.is_global)

class BinOpNode:
	def __init__(self, left_node, operator_token, right_node, start_pos, end_pos):
		self.left_node = left_node
		self.operator_token = operator_token
		self.right_node = right_node

		self.start_pos = start_pos
		self.end_pos = end_pos

	def __repr__(self):
		return f'({self.left_node}, {self.operator_token}, {self.right_node})'
		
	def __hash__(self):
		return hash(self.left_node) ^ hash(self.operator_token) ^ hash(self.right_node)

class UnaryOpNode:
	def __init__(self, operator_token, node, start_pos, end_pos):
		self.operator_token = operator_token
		self.node = node
		
		self.start_pos = start_pos
		self.end_pos = end_pos

	def __repr__(self):
		return f'({self.operator_token}, {self.node})'
		
	def __hash__(self):
		return hash(self.operator_token) ^ hash(self.node)

class IfNode:
	def __init__(self, cases, else_case, should_return_null, start_pos, end_pos):
		self.cases = cases
		self.else_case = else_case
		self.should_return_null = should_return_null

		self.start_pos = start_pos
		self.end_pos = end_pos
		
	def __hash__(self):
		hash_value = hash(0)
		for condition, body in self.cases: hash_value ^= hash(condition) ^ hash(body)

		return hash_value ^ hash(self.else_case) ^ hash(self.should_return_null)

class CountNode:
	def __init__(self, var_name_token, start_value_node, end_value_node, step_value_node, body_node, should_return_null, start_pos, end_pos):
		self.var_name_token = var_name_token
		self.start_value_node = start_value_node
		self.end_value_node = end_value_node
		self.step_value_node = step_value_node
		self.body_node = body_node
		self.should_return_null = should_return_null

		self.start_pos = start_pos
		self.end_pos = end_pos
		
	def __hash__(self):
		return hash(self.var_name_token) ^ hash(self.start_value_node) ^ hash(self.end_value_node) ^ hash(self.step_value_node) ^ hash(self.body_node) ^ hash(self.should_return_null)

class WhileNode:
	def __init__(self, condition_node, body_node, should_return_null, start_pos, end_pos):
		self.condition_node = condition_node
		self.body_node = body_node
		self.should_return_null = should_return_null

		self.start_pos = start_pos
		self.end_pos = end_pos

	def __hash__(self):
		return hash(self.condition_node) ^ hash(self.body_node) ^ hash(self.should_return_null)

class TryNode:
	def __init__(self, body_node, catches, should_return_null, start_pos, end_pos):
		self.body_node = body_node
		self.catches = catches
		self.should_return_null = should_return_null

		self.start_pos = start_pos
		self.end_pos = end_pos
		
	def __hash__(self):
		hash_value = hash(0)
		for error, var_name, body in self.catches: hash_value ^= hash(error) ^ hash(var_name) ^ hash(body)

		return hash_value ^ hash(self.body_node) ^ hash(self.should_return_null)

class FuncDefNode:
	def __init__(self, var_name_token, arg_name_tokens, body_node, should_return_null, start_pos, end_pos):
		self.var_name_token = var_name_token
		self.arg_name_tokens = arg_name_tokens
		self.body_node = body_node
		self.should_return_null = should_return_null

		self.start_pos = start_pos
		self.end_pos = end_pos

	def __hash__(self):
		hash_value = hash(0)
		for i in self.arg_name_tokens: hash_value ^= hash(i)

		return hash(self.var_name_token) ^ hash_value ^ hash(self.body_node) ^ hash(self.should_return_null)

class ObjectDefNode:
	def __init__(self, var_name_token, arg_name_tokens, body_node, start_pos, end_pos):
		self.var_name_token = var_name_token
		self.arg_name_tokens = arg_name_tokens
		self.body_node = body_node

		self.start_pos = start_pos
		self.end_pos = end_pos
		
	def __hash__(self):
		hash_value = hash(0)
		for i in self.arg_name_tokens: hash_value ^= hash(i)

		return hash(self.var_name_token) ^ hash_value ^ hash(self.body_node)

class CallNode:
	def __init__(self, node_to_call, arg_nodes, start_pos, end_pos):
		self.node_to_call = node_to_call
		self.arg_nodes = arg_nodes

		self.start_pos = start_pos
		self.end_pos = end_pos

	def __hash__(self):
		hash_values = hash(0)
		for i in self.arg_nodes: hash_values ^= hash(i)

		return hash(self.node_to_call) ^ hash_values

class ObjectCallNode:
	def __init__(self, object_node, node_to_call, start_pos, end_pos):
		self.object_node = object_node
		self.node_to_call = node_to_call

		self.start_pos = start_pos
		self.end_pos = end_pos

	def __hash__(self):
		return hash(self.object_node) ^ hash(self.node_to_call)

class IncludeNode:
	def __init__(self, name_node, nickname_node, start_pos, end_pos):
		self.name_node = name_node
		self.nickname_node = nickname_node

		self.start_pos = start_pos
		self.end_pos = end_pos

	def __hash__(self):
		return hash(self.name_node) ^ hash(self.nickname_node)

class ReturnNode:
	def __init__(self, node_to_return, start_pos, end_pos):
		self.node_to_return = node_to_return

		self.start_pos = start_pos
		self.end_pos = end_pos

	def __hash__(self):
		return hash(self.node_to_return)

class SkipNode:
	def __init__(self, start_pos, end_pos):
		self.start_pos = start_pos
		self.end_pos = end_pos

	def __hash__(self):
		return hash('SkipNode')

class StopNode:
	def __init__(self, start_pos, end_pos):
		self.start_pos = start_pos
		self.end_pos = end_pos

	def __hash__(self):
		return hash('StopNode')

# PARSE RESULT

class ParseResult:
	def __init__(self):
		self.error = None
		self.node = None
		self.advance_count = 0
		self.to_reverse_count = 0

	def register_advance(self):
		self.advance_count += 1

	def register(self, res):
		self.advance_count += res.advance_count
		if res.error: self.error = res.error
		return res.node

	def try_register(self, res):
		if res.error:
			self.to_reverse_count = res.advance_count
			return None
		return self.register(res)

	def success(self, node):
		self.node = node
		return self

	def failure(self, error):
		if not self.error or self.advance_count == 0:
			self.error = error
		return self

# PARSER

class Parser:
	def __init__(self, tokens):
		self.tokens = tokens
		self.token_idx = -1
		self.current_token = None
		self.advance()

	def advance(self):
		self.token_idx += 1
		self.update_current_token()
		return self.current_token

	def reverse(self, amount=1):
		self.token_idx -= amount
		self.update_current_token()
		return self.current_token

	def update_current_token(self):
		if self.token_idx >= 0 and self.token_idx < len(self.tokens):
			self.current_token = self.tokens[self.token_idx]

	def parse(self):
		res = self.statements()
		if not res.error and self.current_token.type != TT_EOF: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [INT], [FLOAT], [STRING], [IDENTIFIER], 'എങ്കിൽ', 'എണ്ണുക', 'എന്നാൽ', 'ശ്രമിക്കുക', 'പ്രവർത്തനം', 'വസ്തു', 'ഉൾപ്പെടുന്നു', 'വിപരീതം', 'ലോകം', 'ഇനം', 'കൊടുക്കുക', 'തുടരുക', 'നിർത്തൂ', '(', '[', '{', '+' or '-'"))
		return res

	def skip_newlines(self, res):
		newline_count = 0
		while self.current_token.type == TT_NEWLINE:
			res.register_advance()
			self.advance()

			newline_count += 1
		return newline_count

	def statements(self):
		res = ParseResult()
		statements = []
		start_pos = self.current_token.start_pos.copy()
		self.skip_newlines(res)
		
		statement = res.register(self.statement())
		if res.error: return res
		statements.append(statement)

		more_statements = True
		while True:
			newline_count = self.skip_newlines(res)
			if (newline_count == 0
			 	or self.current_token.matches(TT_KEY, 'അവസാനം')
			 	or self.current_token.matches(TT_KEY, 'വേറെ')
			 	or self.current_token.matches(TT_KEY, 'പിശക്')
				or self.current_token.type == TT_EOF): more_statements = False
			if not more_statements: break

			statement = res.register(self.statement())
			if res.error: return res
			
			statements.append(statement)

		return res.success(ListNode(statements, start_pos, self.current_token.end_pos.copy()))

	def statement(self):
		res = ParseResult()
		start_pos = self.current_token.start_pos.copy()

		if self.current_token.matches(TT_KEY, 'കൊടുക്കുക'):
			res.register_advance()
			self.advance()

			expression = res.try_register(self.expression())
			if not expression: self.reverse(res.to_reverse_count)
			return res.success(ReturnNode(expression, start_pos, self.current_token.start_pos.copy()))
		elif self.current_token.matches(TT_KEY, 'തുടരുക'):
			res.register_advance()
			self.advance()
			return res.success(SkipNode(start_pos, self.current_token.start_pos.copy()))
		elif self.current_token.matches(TT_KEY, 'നിർത്തൂ'):
			res.register_advance()
			self.advance()
			return res.success(StopNode(start_pos, self.current_token.start_pos.copy()))
		
		expression = res.register(self.expression())
		if res.error: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [INT], [FLOAT], [STRING], [IDENTIFIER], 'എങ്കിൽ', 'എണ്ണുക', 'എന്നാൽ', 'ശ്രമിക്കുക', 'പ്രവർത്തനം', 'വസ്തു', 'ഉൾപ്പെടുന്നു', 'വിപരീതം', 'ലോകം', 'ഇനം', 'കൊടുക്കുക', 'തുടരുക', 'നിർത്തൂ', '(', '[', '{', '+' or '-'"))
		return res.success(expression)

	def expression(self):
		res = ParseResult()
		start_pos = self.current_token.start_pos.copy()

		is_global = False
		if self.current_token.matches(TT_KEY, 'ലോകം'):
			is_global = True
			res.register_advance()
			self.advance()

		if self.current_token.matches(TT_KEY, 'ഇനം'):
			res.register_advance()
			self.advance()

			if self.current_token.type != TT_ID: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [IDENTIFIER]"))

			var_name = self.current_token
			res.register_advance()
			self.advance()

			if self.current_token.type != TT_COLON: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected ':'"))
			res.register_advance()
			self.advance()

			expression = res.register(self.expression())
			if res.error: return res
			return res.success(VarAssignNode(var_name, expression, is_global, start_pos, self.current_token.end_pos.copy()))
		elif is_global: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'ഇനം'"))
		
		node = res.register(self.binary_operation(self.comp_expr, ((TT_KEY, 'ഉം'), (TT_KEY, 'അല്ലെങ്കിൽ'))))
		if res.error: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [INT], [FLOAT], [STRING], [IDENTIFIER], 'എങ്കിൽ', 'എണ്ണുക', 'എന്നാൽ', 'ശ്രമിക്കുക', 'പ്രവർത്തനം', 'വസ്തു', 'ഉൾപ്പെടുന്നു', 'വിപരീതം', 'ലോകം', 'ഇനം', '(', '[', '{', '+' or '-'"))
		return res.success(node)

	def comp_expr(self):
		res = ParseResult()
		start_pos = self.current_token.start_pos.copy()

		if self.current_token.matches(TT_KEY, 'വിപരീതം'):
			operator_token = self.current_token
			res.register_advance()
			self.advance()

			node = res.register(self.comp_expr())
			if res.error: return res
			return res.success(UnaryOpNode(operator_token, node, start_pos, self.current_token.end_pos.copy()))
		
		node = res.register(self.binary_operation(self.arith_expr, (TT_IE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))
		if res.error: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [INT], [FLOAT], [STRING], [IDENTIFIER], 'എങ്കിൽ', 'എണ്ണുക', 'എന്നാൽ', 'ശ്രമിക്കുക', 'പ്രവർത്തനം', 'വസ്തു', 'ഉൾപ്പെടുന്നു', 'വിപരീതം', '(', '[', '{', '+' or '-'"))
		return res.success(node)

	def arith_expr(self):
		return self.binary_operation(self.term, (TT_PLUS, TT_MINUS))

	def term(self):
		return self.binary_operation(self.factor, (TT_MUL, TT_DIV, TT_MOD))

	def factor(self):
		res = ParseResult()
		start_pos = self.current_token.start_pos.copy()
		token = self.current_token

		if token.type in (TT_PLUS, TT_MINUS):
			res.register_advance()
			self.advance()
			factor = res.register(self.factor())
			if res.error: return res

			return res.success(UnaryOpNode(token, factor, start_pos, self.current_token.end_pos.copy()))

		return self.power()
	
	def power(self):
		return self.binary_operation(self.in_expr, (TT_POW, ), self.factor)

	def in_expr(self):
		return self.binary_operation(self.object_call, ((TT_KEY, 'ൽ'), ), self.object_call)

	def object_call(self):
		res = ParseResult()
		start_pos = self.current_token.start_pos.copy()
		call = res.register(self.call())
		if res.error: return res

		if self.current_token.type == TT_DOT:
			res.register_advance()
			self.advance()

			node_to_call = res.register(self.object_call())
			if res.error: return res

			return res.success(ObjectCallNode(call, node_to_call, start_pos, self.current_token.end_pos.copy()))
		
		return res.success(call)

	def call(self):
		res = ParseResult()
		start_pos = self.current_token.start_pos.copy()
		atom = res.register(self.atom())
		if res.error: return res

		if self.current_token.type == TT_LPAREN:
			res.register_advance()
			self.advance()
			arg_nodes = []

			if self.current_token.type == TT_RPAREN:
				res.register_advance()
				self.advance()
			else:
				arg_nodes.append(res.register(self.expression()))
				if res.error: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [INT], [FLOAT], [STRING], [IDENTIFIER], 'എങ്കിൽ', 'എണ്ണുക', 'എന്നാൽ', 'ശ്രമിക്കുക', 'പ്രവർത്തനം', 'വസ്തു', 'ഉൾപ്പെടുന്നു', 'വിപരീതം', 'ലോകം', 'ഇനം', '(', '[', '{', '+', '-' or ')'"))

				while self.current_token.type == TT_COMMA:
					res.register_advance()
					self.advance()

					arg_nodes.append(res.register(self.expression()))
					if res.error: return res
				
				if self.current_token.type != TT_RPAREN: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected ',' or ')'"))
				res.register_advance()
				self.advance()
		
			return res.success(CallNode(atom, arg_nodes, start_pos, self.current_token.start_pos.copy()))
		return res.success(atom)

	def atom(self):
		res = ParseResult()
		start_pos = self.current_token.start_pos.copy()
		token = self.current_token

		if token.type in (TT_INT, TT_FLOAT):
			res.register_advance()
			self.advance()
			return res.success(NumberNode(token, start_pos, self.current_token.start_pos.copy()))
		elif token.type == TT_STRING:
			res.register_advance()
			self.advance()
			return res.success(StringNode(token, start_pos, self.current_token.start_pos.copy()))
		elif token.type == TT_ID:
			res.register_advance()
			self.advance()
			return res.success(VarAccessNode(token, start_pos, self.current_token.start_pos.copy()))
		elif token.type == TT_LPAREN:
			res.register_advance()
			self.advance()
			expression = res.try_register(self.expression())
			if not expression or self.current_token.type == TT_COMMA:
				self.reverse(res.advance_count)

				array_expression = res.register(self.array_expr())
				if res.error: return res
				return res.success(array_expression)
			
			if self.current_token.type == TT_RPAREN:
				res.register_advance()
				self.advance()
				return res.success(expression)
			return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected ')'"))
		elif token.type == TT_LSQUARE:
			list_expression = res.register(self.list_expr())
			if res.error: return res
			return res.success(list_expression)
		elif token.type == TT_LCURLY:
			dict_expression = res.register(self.dict_expr())
			if res.error: return res
			return res.success(dict_expression)
		elif token.matches(TT_KEY, 'എങ്കിൽ'):
			if_expression = res.register(self.if_expr()[0])
			if res.error: return res
			return res.success(if_expression)
		elif token.matches(TT_KEY, 'എണ്ണുക'):
			count_expression = res.register(self.count_expr())
			if res.error: return res
			return res.success(count_expression)
		elif token.matches(TT_KEY, 'എന്നാൽ'):
			while_expression = res.register(self.while_expr())
			if res.error: return res
			return res.success(while_expression)
		elif token.matches(TT_KEY, 'ശ്രമിക്കുക'):
			try_expression = res.register(self.try_expr())
			if res.error: return res
			return res.success(try_expression)
		elif token.matches(TT_KEY, 'പ്രവർത്തനം'):
			function_expression = res.register(self.function_def())
			if res.error: return res
			return res.success(function_expression)
		elif token.matches(TT_KEY, 'വസ്തു'):
			object_expression = res.register(self.object_def())
			if res.error: return res
			return res.success(object_expression)
		elif token.matches(TT_KEY, 'ഉൾപ്പെടുന്നു'):
			include_expression = res.register(self.include_expr())
			if res.error: return res
			return res.success(include_expression)
			
		return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [INT], [FLOAT], [STRING], [IDENTIFIER], 'എങ്കിൽ', 'എണ്ണുക', 'എന്നാൽ', 'ശ്രമിക്കുക', 'പ്രവർത്തനം', 'വസ്തു', 'ഉൾപ്പെടുന്നു', '(', '[' or '{'"))

	def array_expr(self):
		res = ParseResult()
		element_nodes = []
		start_pos = self.current_token.start_pos.copy()

		if self.current_token.type != TT_LPAREN: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected '('"))
		res.register_advance()
		self.advance()

		if self.current_token.type == TT_RPAREN:
			res.register_advance()
			self.advance()
		else:
			self.skip_newlines(res)
				
			element_nodes.append(res.register(self.expression()))
			if res.error: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [INT], [FLOAT], [STRING], [IDENTIFIER], 'എങ്കിൽ', 'എണ്ണുക', 'എന്നാൽ', 'ശ്രമിക്കുക', 'പ്രവർത്തനം', 'വസ്തു', 'ഉൾപ്പെടുന്നു', 'വിപരീതം', 'ലോകം', 'ഇനം', '(', '[', '{', '+', '-' or ')'"))

			more_elements = True
			self.skip_newlines(res)
			while self.current_token.type == TT_COMMA:
				res.register_advance()
				self.advance()
				self.skip_newlines(res)

				element = res.try_register(self.expression())
				if not element:
					self.reverse(res.to_reverse_count)
					more_elements = False
					break
				element_nodes.append(element)
				
			self.skip_newlines(res)
			if self.current_token.type != TT_RPAREN:
				if more_elements: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected ',' or ')'"))
				return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [INT], [FLOAT], [STRING], [IDENTIFIER], 'എങ്കിൽ', 'എണ്ണുക', 'എന്നാൽ', 'ശ്രമിക്കുക', 'പ്രവർത്തനം', 'വസ്തു', 'ഉൾപ്പെടുന്നു', 'വിപരീതം', 'ലോകം', 'ഇനം', '(', '[', '{', '+', '-', or ')'"))
			
			res.register_advance()
			self.advance()

		return res.success(ArrayNode(element_nodes, start_pos, self.current_token.start_pos.copy()))

	def list_expr(self):
		res = ParseResult()
		element_nodes = []
		start_pos = self.current_token.start_pos.copy()

		if self.current_token.type != TT_LSQUARE: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected '['"))
		res.register_advance()
		self.advance()

		if self.current_token.type == TT_RSQUARE:
			res.register_advance()
			self.advance()
		else:
			self.skip_newlines(res)
				
			element_nodes.append(res.register(self.expression()))
			if res.error: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [INT], [FLOAT], [STRING], [IDENTIFIER], 'എങ്കിൽ', 'എണ്ണുക', 'എന്നാൽ', 'ശ്രമിക്കുക', 'പ്രവർത്തനം', 'വസ്തു', 'ഉൾപ്പെടുന്നു', 'വിപരീതം', 'ലോകം', 'ഇനം', '(', '[', '{', '+', '-' or ']'"))

			self.skip_newlines(res)
			while self.current_token.type == TT_COMMA:
				res.register_advance()
				self.advance()
				self.skip_newlines(res)

				element_nodes.append(res.register(self.expression()))
				if res.error: return res
				
			self.skip_newlines(res)
			if self.current_token.type != TT_RSQUARE: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected ',' or ']'"))
			res.register_advance()
			self.advance()

		return res.success(ListNode(element_nodes, start_pos, self.current_token.start_pos.copy()))

	def dict_expr(self):
		res = ParseResult()
		pair_nodes = []
		start_pos = self.current_token.start_pos.copy()

		if self.current_token.type != TT_LCURLY: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected '{'"))
		res.register_advance()
		self.advance()

		if self.current_token.type == TT_RCURLY:
			res.register_advance()
			self.advance()
		else:
			self.skip_newlines(res)
				
			key = res.register(self.expression())
			if res.error: return res

			if self.current_token.type != TT_COLON: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected ':'"))
			res.register_advance()
			self.advance()

			value = res.register(self.expression())
			if res.error: return res
			pair_nodes.append([key, value])

			self.skip_newlines(res)
			while self.current_token.type == TT_COMMA:
				res.register_advance()
				self.advance()
				self.skip_newlines(res)

				key = res.register(self.expression())
				if res.error: return res

				if self.current_token.type != TT_COLON: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected ':'"))
				res.register_advance()
				self.advance()

				value = res.register(self.expression())
				if res.error: return res
				pair_nodes.append((key, value))
				
			self.skip_newlines(res)
			if self.current_token.type != TT_RCURLY: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected ',' or '}'"))
			res.register_advance()
			self.advance()

		return res.success(DictNode(pair_nodes, start_pos, self.current_token.start_pos.copy()))

	def if_expr(self, secondary_call=False):
		res = ParseResult()
		start_pos = self.current_token.start_pos.copy()

		cases = []
		else_case = None
		has_ended = False

		if not self.current_token.matches(TT_KEY, 'എങ്കിൽ'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'എങ്കിൽ'")), None, has_ended
		res.register_advance()
		self.advance()

		condition = res.register(self.expression())
		if res.error: return res, None, has_ended

		if not self.current_token.matches(TT_KEY, 'ചെയ്യുക'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'ചെയ്യുക'")), None, has_ended
		res.register_advance()
		self.advance()

		if self.current_token.type == TT_NEWLINE:
			res.register_advance()
			self.advance()

			statements = res.register(self.statements())
			if res.error: return res, None, has_ended
			cases.append((condition, statements))

			if self.current_token.matches(TT_KEY, 'അവസാനം'):
				has_ended = True
				res.register_advance()
				self.advance()
			else:
				if not self.current_token.matches(TT_KEY, 'വേറെ'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'വേറെ' or 'അവസാനം'")), None
				res.register_advance()
				self.advance()
            
				if self.current_token.matches(TT_KEY, 'എങ്കിൽ'):
					if_expr, added_cases, has_ended = self.if_expr(secondary_call=True)
					res.register(if_expr)
					if res.error: return res, None, has_ended
					cases.extend(added_cases)
            
				if not has_ended and not secondary_call:
					if self.tokens[self.token_idx-1].matches(TT_KEY, 'വേറെ'):
						self.reverse()

					if self.current_token.matches(TT_KEY, 'വേറെ'):
						res.register_advance()
						self.advance()
	
						if not self.current_token.matches(TT_KEY, 'ചെയ്യുക'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'ചെയ്യുക'")), None, has_ended
						res.register_advance()
						self.advance()
	
						else_case = res.register(self.statements())
						if res.error: return res, None, has_ended

						if not self.current_token.matches(TT_KEY, 'അവസാനം'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'വേറെ' or 'അവസാനം'")), None, has_ended
						res.register_advance()
						self.advance()
					elif not self.current_token.matches(TT_KEY, 'വേറെ'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'വേറെ' or 'അവസാനം'")), None, has_ended
			return res.success(IfNode(cases, else_case, True, start_pos, self.current_token.end_pos.copy())), cases, has_ended
		else:
			expression = res.register(self.statement())
			if res.error: return res, None, has_ended
			cases.append((condition, expression))

			if not secondary_call:
				while self.current_token.matches(TT_KEY, 'വേറെ'):
					res.register_advance()
					self.advance()

					if self.current_token.matches(TT_KEY, 'if'):
						if_expr, added_cases, has_ended = self.if_expr(secondary_call=True)
						res.register(if_expr)
						if res.error: return res, None, has_ended
						cases.extend(added_cases)
					else:
						if not self.current_token.matches(TT_KEY, 'ചെയ്യുക'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'ചെയ്യുക'")), None, has_ended
						res.register_advance()
						self.advance()

						else_case = res.register(self.statement())
						if res.error: return res, None, has_ended

			return res.success(IfNode(cases, else_case, False, start_pos, self.current_token.end_pos.copy())), cases, has_ended

	def count_expr(self):
		res = ParseResult()
		start_pos = self.current_token.start_pos.copy()

		if not self.current_token.matches(TT_KEY, 'എണ്ണുക'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'എണ്ണുക'"))
		res.register_advance()
		self.advance()

		if not self.current_token.matches(TT_KEY, 'നിന്ന്'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'നിന്ന്'"))
		res.register_advance()
		self.advance()

		start_value = res.register(self.expression())
		if res.error: return res

		if not self.current_token.matches(TT_KEY, 'പോലെ'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'പോലെ'"))
		res.register_advance()
		self.advance()

		if self.current_token.type != TT_ID: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [IDENTIFIER]"))
		var_name = self.current_token
		res.register_advance()
		self.advance()

		if not self.current_token.matches(TT_KEY, 'വരെ'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'വരെ'"))
		res.register_advance()
		self.advance()

		end_value = res.register(self.expression())
		if res.error: return res

		if self.current_token.matches(TT_KEY, 'ഘട്ടം'):
			res.register_advance()
			self.advance()

			step_value = res.register(self.expression())
			if res.error: return res
		else:
			step_value = None
		
		if not self.current_token.matches(TT_KEY, 'ചെയ്യുക'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'ചെയ്യുക'"))
		res.register_advance()
		self.advance()

		if self.current_token.type == TT_NEWLINE:
			res.register_advance()
			self.advance()
			
			body = res.register(self.statements())
			if res.error: return res
			
			if not self.current_token.matches(TT_KEY, 'അവസാനം'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'അവസാനം'"))
			res.register_advance()
			self.advance()
			
			return res.success(CountNode(var_name, start_value, end_value, step_value, body, True, start_pos, self.current_token.start_pos.copy()))
		else:
			body = res.register(self.statement())
			if res.error: return res
			return res.success(CountNode(var_name, start_value, end_value, step_value, body, False, start_pos, self.current_token.end_pos.copy()))

	def while_expr(self):
		res = ParseResult()
		start_pos = self.current_token.start_pos.copy()

		if not self.current_token.matches(TT_KEY, 'എന്നാൽ'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'എന്നാൽ'"))
		res.register_advance()
		self.advance()

		condition = res.register(self.expression())
		if res.error: return res

		if not self.current_token.matches(TT_KEY, 'ചെയ്യുക'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'ചെയ്യുക'"))
		res.register_advance()
		self.advance()

		if self.current_token.type == TT_NEWLINE:
			res.register_advance()
			self.advance()

			body = res.register(self.statements())
			if res.error: return res

			if not self.current_token.matches(TT_KEY, 'അവസാനം'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'അവസാനം'"))
			res.register_advance()
			self.advance()
			
			return res.success(WhileNode(condition, body, True, start_pos, self.current_token.start_pos.copy()))
		else:
			body = res.register(self.statement())
			if res.error: return res
			return res.success(WhileNode(condition, body, False, start_pos, self.current_token.end_pos.copy()))

	def try_expr(self):
		res = ParseResult()
		start_pos = self.current_token.start_pos
		body_node = None
		catches = []

		if not self.current_token.matches(TT_KEY, 'ശ്രമിക്കുക'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'ശ്രമിക്കുക'"))
		res.register_advance()
		self.advance()

		if not self.current_token.matches(TT_KEY, 'ചെയ്യുക'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'ചെയ്യുക'"))
		res.register_advance()
		self.advance()

		if self.current_token.type == TT_NEWLINE:
			res.register_advance()
			self.advance()

			statements = res.register(self.statements())
			if res.error: return res
			body_node = statements

			empty_errors = []
			while self.current_token.matches(TT_KEY, 'പിശക്'):
				res.register_advance()
				self.advance()

				error = None
				var_name = None
				if self.current_token.type == TT_STRING:
					error = self.current_token
					res.register_advance()
					self.advance()

				if self.current_token.matches(TT_KEY, 'പോലെ'):
					res.register_advance()
					self.advance()

					if self.current_token.type != TT_ID: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [IDENTIFIER]"))
					
					var_name = self.current_token
					res.register_advance()
					self.advance()

				if not self.current_token.matches(TT_KEY, 'ചെയ്യുക'):
					if error == None and var_name == None: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [STRING], 'പോലെ' or 'ചെയ്യുക'"))
					elif error != None and var_name == None: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'പോലെ' or 'ചെയ്യുക'"))
					return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'ചെയ്യുക'"))
				res.register_advance()
				self.advance()

				expression = res.register(self.statements())
				if res.error: return res

				if error == None: empty_errors.append(len(catches))
				catches.append((error, var_name, expression))

			if len(empty_errors) != 0:
				if len(empty_errors) > 1: return res.failure(InvalidSyntaxError(start_pos, self.current_token.end_pos, "There cannot be more than one empty 'പിശക്' statement"))
				if empty_errors[0] != len(catches)-1: return res.failure(InvalidSyntaxError(start_pos, self.current_token.end_pos, "The empty 'പിശക്' statement should always be declared last"))

			if not self.current_token.matches(TT_KEY, 'അവസാനം'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'അവസാനം'"))
			res.register_advance()
			self.advance()

			return res.success(TryNode(body_node, catches, True, start_pos, self.current_token.start_pos.copy()))
		else:
			expression = res.register(self.statement())
			if res.error: return res
			body_node = expression

			empty_errors = []
			while self.current_token.matches(TT_KEY, 'പിശക്'):
				res.register_advance()
				self.advance()

				error = None
				var_name = None
				if self.current_token.type == TT_STRING:
					error = self.current_token
					res.register_advance()
					self.advance()

				if self.current_token.matches(TT_KEY, 'പോലെ'):
					res.register_advance()
					self.advance()

					if self.current_token.type != TT_ID: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [IDENTIFIER]"))
					
					var_name = self.current_token
					res.register_advance()
					self.advance()

				if not self.current_token.matches(TT_KEY, 'ചെയ്യുക'):
					if error == None and var_name == None: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [STRING], 'പോലെ' or 'ചെയ്യുക'"))
					elif error != None and var_name == None: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'പോലെ' or 'ചെയ്യുക'"))
					return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'ചെയ്യുക'"))
				res.register_advance()
				self.advance()

				expression = res.register(self.statement())
				if res.error: return res

				if error == None: empty_errors.append(len(catches))
				catches.append((error, var_name, expression))
			
			if len(empty_errors) != 0:
				if len(empty_errors) > 1: return res.failure(InvalidSyntaxError(start_pos, self.current_token.end_pos, "There cannot be more than one empty 'പിശക്' statement"))
				if empty_errors[0] != len(catches)-1: return res.failure(InvalidSyntaxError(start_pos, self.current_token.end_pos, "The empty 'പിശക്' statement should always be declared last"))

			return res.success(TryNode(body_node, catches, False, start_pos, self.current_token.end_pos.copy()))

	def function_def(self):
		res = ParseResult()
		start_pos = self.current_token.start_pos.copy()

		if not self.current_token.matches(TT_KEY, 'പ്രവർത്തനം'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'പ്രവർത്തനം'"))
		res.register_advance()
		self.advance()

		if self.current_token.type == TT_ID:
			var_name = self.current_token
			res.register_advance()
			self.advance()
		else:
			var_name = None

		arg_names = []
		if self.current_token.matches(TT_KEY, 'കൂടെ'):
			res.register_advance()
			self.advance()

			if self.current_token.type != TT_ID: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [IDENTIFIER]"))
			arg_names.append(self.current_token)
			res.register_advance()
			self.advance()

			while self.current_token.type == TT_COMMA:
				res.register_advance()
				self.advance()

				if self.current_token.type != TT_ID: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [IDENTIFIER]"))
				arg_names.append(self.current_token)
				res.register_advance()
				self.advance()
		
			if not self.current_token.matches(TT_KEY, 'ചെയ്യുക'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected ',' or 'ചെയ്യുക'"))
		else:
			if not self.current_token.matches(TT_KEY, 'ചെയ്യുക'):
				if var_name == None: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [IDENTIFIER], 'കൂടെ' or 'ചെയ്യുക'"))
				return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'കൂടെ' or 'ചെയ്യുക'"))
		res.register_advance()
		self.advance()

		if self.current_token.type == TT_NEWLINE:
			node_to_return = res.register(self.statements())
			if res.error: return res

			if not self.current_token.matches(TT_KEY, 'അവസാനം'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'അവസാനം'"))
			res.register_advance()
			self.advance()

			return res.success(FuncDefNode(var_name, arg_names, node_to_return, True, start_pos, self.current_token.start_pos.copy()))
		else:
			node_to_return = res.register(self.expression())
			if res.error: return res

			return res.success(FuncDefNode(var_name, arg_names, node_to_return, False, start_pos, self.current_token.end_pos.copy()))

	def object_def(self):
		res = ParseResult()
		start_pos = self.current_token.start_pos.copy()

		if not self.current_token.matches(TT_KEY, 'വസ്തു'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'വസ്തു'"))
		res.register_advance()
		self.advance()

		if self.current_token.type != TT_ID: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [IDENTIFIER]"))
		var_name = self.current_token
		res.register_advance()
		self.advance()

		arg_names = []
		if self.current_token.matches(TT_KEY, 'കൂടെ'):
			res.register_advance()
			self.advance()

			if self.current_token.type != TT_ID: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [IDENTIFIER]"))
			arg_names.append(self.current_token)
			res.register_advance()
			self.advance()

			while self.current_token.type == TT_COMMA:
				res.register_advance()
				self.advance()

				if self.current_token.type != TT_ID: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, 'Expected [IDENTIFIER]'))
				arg_names.append(self.current_token)
				res.register_advance()
				self.advance()
		
			if not self.current_token.matches(TT_KEY, 'ചെയ്യുക'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected ',' or 'ചെയ്യുക'"))
		else:
			if not self.current_token.matches(TT_KEY, 'ചെയ്യുക'):
				if var_name == None: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [IDENTIFIER], 'കൂടെ' or 'ചെയ്യുക'"))
				return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'കൂടെ' or 'ചെയ്യുക'"))
		res.register_advance()
		self.advance()

		if self.current_token.type == TT_NEWLINE:
			node_to_return = res.register(self.statements())
			if res.error: return res

			if not self.current_token.matches(TT_KEY, 'അവസാനം'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'അവസാനം'"))
			res.register_advance()
			self.advance()
			
			return res.success(ObjectDefNode(var_name, arg_names, node_to_return, start_pos, self.current_token.start_pos.copy()))
		else:
			node_to_return = res.register(self.expression())
			if res.error: return res

			return res.success(ObjectDefNode(var_name, arg_names, node_to_return, start_pos, self.current_token.end_pos.copy()))

	def include_expr(self):
		res = ParseResult()
		start_pos = self.current_token.start_pos.copy()

		if not self.current_token.matches(TT_KEY, 'ഉൾപ്പെടുന്നു'): return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected 'ഉൾപ്പെടുന്നു'"))
		res.register_advance()
		self.advance()

		if self.current_token.type != TT_STRING: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [STRING]"))
		name = self.current_token
		res.register_advance()
		self.advance()

		nickname = None
		if self.current_token.matches(TT_KEY, 'പോലെ'):
			res.register_advance()
			self.advance()

			if self.current_token.type != TT_ID: return res.failure(InvalidSyntaxError(self.current_token.start_pos, self.current_token.end_pos, "Expected [IDENTIFIER]"))
			nickname = self.current_token
			res.register_advance()
			self.advance()
		
		return res.success(IncludeNode(name, nickname, start_pos, self.current_token.start_pos.copy()))
			
	def binary_operation(self, func_a, ops, func_b=None):
		if func_b == None: func_b = func_a

		res = ParseResult()
		start_pos = self.current_token.start_pos.copy()

		left = res.register(func_a())
		if res.error: return res

		while self.current_token.type in ops or (self.current_token.type, self.current_token.value) in ops:
			operator_token = self.current_token
			res.register_advance()
			self.advance()
			right = res.register(func_b())
			if res.error: return res

			left = BinOpNode(left, operator_token, right, start_pos, self.current_token.end_pos.copy())

		return res.success(left)

# RUNTIME RESULT

class RuntimeResult:
	def __init__(self):
		self.reset()

	def reset(self):
		self.value = None
		self.error = None
		self.function_return_value = None
		self.loop_should_skip = False
		self.loop_should_stop = False

	def register(self, res):
		if res.error: self.error = res.error
		self.function_return_value = res.function_return_value
		self.loop_should_skip = res.loop_should_skip
		self.loop_should_stop = res.loop_should_stop
		return res.value

	def success(self, value):
		self.reset()
		self.value = value
		return self
	
	def success_return(self, value):
		self.reset()
		self.function_return_value = value
		return self
	
	def success_skip(self):
		self.reset()
		self.loop_should_skip = True
		return self
	
	def success_stop(self):
		self.reset()
		self.loop_should_stop = True
		return self

	def failure(self, error):
		self.reset()
		self.error = error
		return self

	def should_return(self):
		return (self.error or self.function_return_value or self.loop_should_skip or self.loop_should_stop)

# VALUES

class Value:
	def __init__(self):
		self.set_pos()
		self.set_context()

	def set_pos(self, start_pos=None, end_pos=None):
		self.start_pos = start_pos
		self.end_pos = end_pos
		return self

	def set_context(self, context=None):
		self.context = context
		return self
	
	def added_to(self, other):
		return None, self.illegal_operation(other)

	def subbed_by(self, other):
		return None, self.illegal_operation(other)
		
	def multed_by(self, other):
		return None, self.illegal_operation(other)

	def dived_by(self, other):
		return None, self.illegal_operation(other)

	def moded_by(self, other):
		return None, self.illegal_operation(other)

	def powed_by(self, other):
		return None, self.illegal_operation(other)
		
	def compare_eq(self, other):
		return None, self.illegal_operation(other)
		
	def compare_ne(self, other):
		return None, self.illegal_operation(other)
		
	def compare_lt(self, other):
		return None, self.illegal_operation(other)
		
	def compare_gt(self, other):
		return None, self.illegal_operation(other)
		
	def compare_lte(self, other):
		return None, self.illegal_operation(other)
		
	def compare_gte(self, other):
		return None, self.illegal_operation(other)
	
	def compare_and(self, other):
		return None, self.illegal_operation(other)
		
	def compare_or(self, other):
		return None, self.illegal_operation(other)
		
	def check_in(self, other):
		return None, self.illegal_operation(other)
	
	def invert(self):
		return None, self.illegal_operation()

	def execute(self, args):
		return RuntimeResult().failure(RuntimeError(self.start_pos, self.end_pos, RTE_ILLEGALOP, f'പിന്തുണയ്‌ക്കാത്ത പ്രവർത്തനം വിളി', self.context))

	def retrieve(self, node):
		return RuntimeResult().failure(RuntimeError(self.start_pos, self.end_pos, RTE_ILLEGALOP, f'പിന്തുണയ്‌ക്കാത്ത പ്രവർത്തനം വിളി', self.context))

	def is_true(self):
		return False

	def copy(self):
		raise Exception('No copy method defined!')

	def illegal_operation(self, other=None):
		otherstr = f' ഉം \'{type(other).__name__}\' ഉം'
		return RuntimeError(self.start_pos, self.end_pos, RTE_ILLEGALOP, f'\'{type(self).__name__}\'{otherstr if other else ""} തരത്തിനായുള്ള നിയമവിരുദ്ധ പ്രവർത്തനം', self.context)

	__hash__ = None

class Bool(Value):
	def __init__(self, value):
		super().__init__()
		self.value = value
		
	def compare_eq(self, other):
		if hasattr(other, 'is_true'): return Bool(other.is_true() == self.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
		
	def compare_ne(self, other):
		if hasattr(other, 'is_true'): return Bool(other.is_true() != self.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def compare_and(self, other):
		if hasattr(other, 'is_true'): return Bool(other.is_true() and self.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
	
	def compare_or(self, other):
		if hasattr(other, 'is_true'): return Bool(other.is_true() or self.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def check_in(self, other):
		if isinstance(other, (Array, List, Dict)): return Bool(other.has_element(self)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def invert(self):
		return Bool(not self.value).set_context(self.context), None
		
	def is_true(self):
		return self.value

	def copy(self):
		copy_ = Bool(self.value)
		copy_.set_pos(self.start_pos, self.end_pos)
		copy_.set_context(self.context)
		return copy_

	def __repr__(self):
		return 'സത്യം' if self.value else 'തെറ്റ്'

	def __hash__(self):
		return hash(self.value)

Bool.true  = Bool(True)
Bool.false = Bool(False)

class Nothing(Value):
	def __init__(self):
		super().__init__()
		
	def compare_eq(self, other):
		return Bool(isinstance(other, Nothing)).set_context(self.context), None
		
	def compare_ne(self, other):
		return Bool(not isinstance(other, Nothing)).set_context(self.context), None

	def compare_and(self, other):
		return Bool(False).set_context(self.context), None
	
	def compare_or(self, other):
		if hasattr(other, 'is_true'): return Bool(other.is_true()).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def check_in(self, other):
		if isinstance(other, (Array, List, Dict)): return Bool(other.has_element(self)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
		
	def is_true(self):
		return False

	def copy(self):
		copy_ = Nothing()
		copy_.set_pos(self.start_pos, self.end_pos)
		copy_.set_context(self.context)
		return copy_

	def __repr__(self):
		return 'ഒന്നുമില്ല'

	def __hash__(self):
		return hash(None)

Nothing.nothing = Nothing()

class Number(Value):
	def __init__(self, value):
		super().__init__()
		self.value = value

	def added_to(self, other):
		if isinstance(other, Number): return Number(self.value + other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def subbed_by(self, other):
		if isinstance(other, Number): return Number(self.value - other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
		
	def multed_by(self, other):
		if isinstance(other, Number): return Number(self.value * other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def dived_by(self, other):
		if isinstance(other, Number): 
			if other.value == 0: return None, RuntimeError(other.start_pos, other.end_pos, RTE_MATH, 'പൂജ്യം കൊണ്ട് ഹരണം', self.context)
			return Number(self.value / other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def moded_by(self, other):
		if isinstance(other, Number): 
			if other.value == 0: return None, RuntimeError(other.start_pos, other.end_pos, RTE_MATH, 'പൂജ്യം കൊണ്ട് മൊഡ്യൂളോ', self.context)
			return Number(self.value % other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def powed_by(self, other):
		if isinstance(other, Number): return Number(self.value ** other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
		
	def compare_eq(self, other):
		if isinstance(other, Number): return Bool(self.value == other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
		
	def compare_ne(self, other):
		if isinstance(other, Number): return Bool(self.value != other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
		
	def compare_lt(self, other):
		if isinstance(other, Number): return Bool(self.value < other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
		
	def compare_gt(self, other):
		if isinstance(other, Number): return Bool(self.value > other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
		
	def compare_lte(self, other):
		if isinstance(other, Number): return Bool(self.value <= other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
		
	def compare_gte(self, other):
		if isinstance(other, Number): return Bool(self.value >= other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
	
	def compare_and(self, other):
		if isinstance(other, Number): return Bool(self.is_true() and other.is_true()).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
		
	def compare_or(self, other):
		if isinstance(other, Number): return Bool(self.is_true() or other.is_true()).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
	
	def check_in(self, other):
		if isinstance(other, (Array, List, Dict)): return Bool(other.has_element(self)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def invert(self):
		return Number(1 if self.value == 0 else 0).set_context(self.context), None

	def is_true(self):
		return self.value != 0

	def copy(self):
		copy_ = Number(self.value)
		copy_.set_pos(self.start_pos, self.end_pos)
		copy_.set_context(self.context)
		return copy_

	def __repr__(self):
		return str(self.value)

	def __hash__(self):
		return hash(self.value)

class String(Value):
	def __init__(self, value):
		super().__init__()
		self.value = value
	
	def added_to(self, other):
		if isinstance(other, String): return String(self.value + other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
	
	def multed_by(self, other):
		if isinstance(other, Number): return String(self.value * int(other.value)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def dived_by(self, other):
		if isinstance(other, Number): 
			if other.value == 0: return None, RuntimeError(other.start_pos, other.end_pos, RTE_MATH, 'പൂജ്യം കൊണ്ട് ഹരണം', self.context)
			return String(self.value[:int(len(self.value)/other.value)]).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
		
	def compare_lte(self, other):
		if isinstance(other, Number):
			try: return String(self.value[int(other.value)]).set_context(self.context), None
			except Exception: return None, RuntimeError(other.start_pos, other.end_pos, RTE_IDXOUTOFRANGE, 'പരിധിക്ക് പുറത്തുള്ള സൂചിക', self.context)
		return None, Value.illegal_operation(self, other)

	def compare_eq(self, other):
		if isinstance(other, String): return Bool(self.value == other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
		
	def compare_ne(self, other):
		if isinstance(other, String): return Bool(self.value != other.value).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
	
	def compare_and(self, other):
		if isinstance(other, String): return Bool(self.is_true() and other.is_true()).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
		
	def compare_or(self, other):
		if isinstance(other, String): return Bool(self.is_true() or other.is_true()).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
	
	def check_in(self, other):
		if isinstance(other, String): return Bool(self.value in other.value).set_context(self.context), None
		if isinstance(other, (Array, List, Dict)): return Bool(other.has_element(self)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def is_true(self):
		return len(self.value) > 0

	def copy(self):
		copy_ = String(self.value)
		copy_.set_pos(self.start_pos, self.end_pos)
		copy_.set_context(self.context)
		return copy_

	def __str__(self):
		return f'{self.value}'

	def __repr__(self):
		return f'\'{repr(self.value)[1:-1]}\''

	def __hash__(self):
		return hash(self.value)

String.ezr_version = String(VERSION)

class List(Value):
	def __init__(self, elements):
		super().__init__()
		self.elements = elements

	def has_element(self, other):
		for i in self.elements:
			if hash(i) == hash(other): return True
		return False

	def added_to(self, other):
		if isinstance(other, List): self.elements.extend(other.elements)
		else: self.elements.append(other)

		return Nothing().set_context(self.context), None
	
	def subbed_by(self, other):
		if isinstance(other, Number):
			try:
				self.elements.pop(int(other.value))
				return Nothing().set_context(self.context), None
			except Exception: return None, RuntimeError(other.start_pos, other.end_pos, RTE_IDXOUTOFRANGE, 'പരിധിക്ക് പുറത്തുള്ള സൂചിക', self.context)
		return None, Value.illegal_operation(self, other)

	def multed_by(self, other):
		if isinstance(other, Number): return List(self.elements * int(other.value)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def dived_by(self, other):
		if isinstance(other, Number): 
			if other.value == 0: return None, RuntimeError(other.start_pos, other.end_pos, RTE_MATH, 'പൂജ്യം കൊണ്ട് ഹരണം', self.context)
			elements = self.elements[:int(len(self.elements)/other.value)]
			return List(elements).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
	
	def compare_lte(self, other):
		if isinstance(other, Number):
			try: return self.elements[int(other.value)], None
			except Exception: return None, RuntimeError(other.start_pos, other.end_pos, RTE_IDXOUTOFRANGE, 'പരിധിക്ക് പുറത്തുള്ള സൂചിക', self.context)
		return None, Value.illegal_operation(self, other)

	def compare_eq(self, other):
		if isinstance(other, List): return Bool(hash(self) == hash(other)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def compare_ne(self, other):
		if isinstance(other, List): return Bool(hash(self) != hash(other)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def compare_and(self, other):
		if isinstance(other, List): return Bool(self.is_true() and other.is_true()).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
	
	def compare_or(self, other):
		if isinstance(other, List): return Bool(self.is_true() or other.is_true()).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def check_in(self, other):
		if isinstance(other, (Array, List, Dict)): return Bool(other.has_element(self)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def is_true(self):
		return len(self.elements) > 0

	def copy(self):
		copy_ = List(self.elements)
		copy_.set_context(self.context)
		copy_.set_pos(self.start_pos, self.end_pos)
		return copy_

	def __repr__(self):
		return f'[{", ".join([repr(i) for i in self.elements])}]'

	def __hash__(self):
		return hash(tuple(self.elements))

class Array(Value):
	def __init__(self, elements):
		super().__init__()
		self.elements = tuple(elements)

	def has_element(self, other):
		for i in self.elements:
			if hash(i) == hash(other): return True
		return False

	def multed_by(self, other):
		if isinstance(other, Number): return Array(self.elements * int(other.value)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def dived_by(self, other):
		if isinstance(other, Number): 
			if other.value == 0: return None, RuntimeError(other.start_pos, other.end_pos, RTE_MATH, 'പൂജ്യം കൊണ്ട് ഹരണം', self.context)
			elements = self.elements[:int(len(self.elements)/other.value)]
			return Array(elements).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def compare_lte(self, other):
		if isinstance(other, Number):
			try: return self.elements[int(other.value)], None
			except Exception: return None, RuntimeError(other.start_pos, other.end_pos, RTE_IDXOUTOFRANGE, 'പരിധിക്ക് പുറത്തുള്ള സൂചിക', self.context)
		return None, Value.illegal_operation(self, other)

	def compare_eq(self, other):
		if isinstance(other, Array): return Bool(hash(self) == hash(other)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def compare_ne(self, other):
		if isinstance(other, Array): return Bool(hash(self) != hash(other)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def compare_and(self, other):
		if isinstance(other, Array): return Bool(self.is_true() and other.is_true()).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
	
	def compare_or(self, other):
		if isinstance(other, Array): return Bool(self.is_true() or other.is_true()).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def check_in(self, other):
		if isinstance(other, (Array, List, Dict)): return Bool(other.has_element(self)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def is_true(self):
		return len(self.elements) > 0

	def copy(self):
		copy_ = Array(self.elements)
		copy_.set_context(self.context)
		copy_.set_pos(self.start_pos, self.end_pos)
		return copy_

	def __repr__(self):
		return f'({", ".join([repr(i) for i in self.elements])})'

	def __hash__(self):
		return hash(self.elements)

class Dict(Value):
	def __init__(self, dict):
		super().__init__()
		self.dict = dict

	def has_element(self, other):
		if hash(other) in self.dict.keys(): return True
		return False

	def added_to(self, other):
		if isinstance(other, Dict):
			self.dict.update(other.dict)
			return Nothing().set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def subbed_by(self, other):
		if other.__hash__:
			if hash(other) in self.dict.keys():
				self.dict.pop(hash(other))
				return Nothing().set_context(self.context), None

			return None, RuntimeError(other.start_pos, other.end_pos, RTE_DICTKEY, f'താക്കോൽ \'{str(other)}\'  നിലവിലില്ല', self.context)
		return None, RuntimeError(other.start_pos, other.end_pos, RTE_DICTKEY, f'താക്കോൽ \'{str(other)}\' ക്രമീകരിക്കാൻ പറ്റില്ല', self.context)

	def dived_by(self, other):
		if isinstance(other, Number): 
			if other.value == 0: return None, RuntimeError(other.start_pos, other.end_pos, RTE_MATH, 'പൂജ്യം കൊണ്ട് ഹരണം', self.context)
			new_dict = {}
			length = int(len(self.dict)/other.value)
			keys = tuple(self.dict.keys())
			for i in range(length): new_dict[keys[i]] = self.dict[keys[i]]
			return Dict(new_dict).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def compare_lte(self, other):
		if other.__hash__:
			if hash(other) in self.dict.keys(): return self.dict[hash(other)][1].set_context(self.context), None

			return None, RuntimeError(other.start_pos, other.end_pos, RTE_DICTKEY, f'താക്കോൽ \'{str(other)}\'  നിലവിലില്ല', self.context)
		return None, RuntimeError(other.start_pos, other.end_pos, RTE_DICTKEY, f'താക്കോൽ \'{str(other)}\' ക്രമീകരിക്കാൻ പറ്റില്ല', self.context)

	def compare_eq(self, other):
		if isinstance(other, Dict): return Bool(hash(self) == hash(other)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def compare_ne(self, other):
		if isinstance(other, Dict): return Bool(hash(self) != hash(other)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def compare_and(self, other):
		if isinstance(other, Dict): return Bool(self.is_true() and other.is_true()).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
	
	def compare_or(self, other):
		if isinstance(other, Dict): return Bool(self.is_true() or other.is_true()).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def check_in(self, other):
		if isinstance(other, (Array, List, Dict)): return Bool(other.has_element(self)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def is_true(self):
		return len(self.dict) > 0

	def copy(self):
		copy_ = Dict(self.dict)
		copy_.set_context(self.context)
		copy_.set_pos(self.start_pos, self.end_pos)
		return copy_
		
	def __repr__(self):
		return '{' + ", ".join([f"{key} : {repr(value)}" for key, value in self.dict.values()]) + '}'

	def __hash__(self):
		hash_value = hash(0)
		for key in self.dict.keys(): hash_value ^= key ^ hash(self.dict[key])
		return hash_value
        
class BaseFunction(Value):
	def __init__(self, name):
		super().__init__()
		self.name = name or '<anonymous>'

	def generate_context(self):
		new_context = Context(self.name, self.context, self.start_pos)
		new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
		return new_context

	def check_args(self, arg_names, args):
		res = RuntimeResult()
		if len(args) > len(arg_names): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_TOOMANYARGS, f'\'{self.name}\' എന്നതിന് {len(args)-len(arg_names)} അധിക വാദങ്ങൾ നൽകി', self.context))
		if len(args) < len(arg_names): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_TOOFEWARGS, f'\'{self.name}\' എന്നതിന് {len(arg_names)-len(args)} വാദങ്ങൾ കൂടി നൽകണം', self.context))

		return res.success(None)
	
	def populate_args(self, arg_names, args, context):
		for i in range(len(args)):
			arg_name = arg_names[i]
			arg_value = args[i]
			arg_value.set_context(context)
			context.symbol_table.set(arg_name, arg_value)

	def check_and_populate_args(self, arg_names, args, context):
		res = RuntimeResult()
		res.register(self.check_args(arg_names, args))
		if res.should_return(): return res
		self.populate_args(arg_names, args, context)
		return res.success(None)

	def compare_eq(self, other):
		if self.__hash__ and other.__hash__: return Bool(hash(self) == hash(other)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def compare_ne(self, other):
		if self.__hash__ and other.__hash__: return Bool(hash(self) != hash(other)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)

	def check_in(self, other):
		if isinstance(other, (Array, List, Dict)): return Bool(other.has_element(self)).set_context(self.context), None
		return None, Value.illegal_operation(self, other)
	
	def is_true(self):
		return True

class Function(BaseFunction):
	def __init__(self, name, body_node, arg_names, should_return_null):
		super().__init__(name)
		self.body_node = body_node
		self.arg_names = arg_names
		self.should_return_null = should_return_null

	def execute(self, args):
		res = RuntimeResult()
		interpreter = Interpreter()
		new_context = self.generate_context()
		res.register(self.check_and_populate_args(self.arg_names, args, new_context))
		if res.should_return(): return res

		value = res.register(interpreter.visit(self.body_node, new_context))
		if res.should_return() and res.function_return_value == None: return res

		return_value = (None if self.should_return_null else value) or res.function_return_value or Nothing()
		return res.success(return_value)
	
	def copy(self):
		copy_ = Function(self.name, self.body_node, self.arg_names, self.should_return_null)
		copy_.set_context(self.context)
		copy_.set_pos(self.start_pos, self.end_pos)
		return copy_
	
	def __repr__(self):
		return f'<പ്രവർത്തനം <{self.name}>>'

	def __hash__(self):
		hash_value = hash(0)
		for i in self.arg_names: hash_value ^= hash(i)

		return hash(self.body_node) ^ hash(self.should_return_null) ^ hash(self.name) ^ hash_value

class BuiltInFunction(BaseFunction):
	def __init__(self, name):
		super().__init__(name)

	def execute(self, args):
		res = RuntimeResult()
		context = self.generate_context()

		method_name = f'execute_{self.name}'
		method = getattr(self, method_name, self.no_visit_method)
		res.register(self.check_and_populate_args(method.arg_names, args, context))
		if res.should_return(): return res

		return_value = res.register(method(context))
		if res.should_return(): return res
		return res.success(return_value)

	def no_visit_method(self, node, context):
		raise Exception(f'No execute_{self.name} method defined!')

	def execute_കാണിക്കുക(self, context):
		print(str(context.symbol_table.get('out')))
		return RuntimeResult().success(Nothing())
	execute_കാണിക്കുക.arg_names = ['out']

	def execute_പിശക്_കാണിക്കുക(self, context):
		return RuntimeResult().failure(RuntimeError(self.start_pos, self.end_pos, RTE_CUSTOM, str(context.symbol_table.get('out')), context.parent))
	execute_പിശക്_കാണിക്കുക.arg_names = ['out']
	
	def execute_നേടുക(self, context):
		return RuntimeResult().success(String(input(str(context.symbol_table.get('out')))))
	execute_നേടുക.arg_names = ['out']
	
	def execute_പൂർണ്ണസംഖ്യ_നേടുക(self, context):
		while True:
			text = input(str(context.symbol_table.get('out')))
			try: number = int(text); break
			except ValueError: print(f'\'{text}\' must be an [INT]. Try again!')

		return RuntimeResult().success(Number(number))
	execute_പൂർണ്ണസംഖ്യ_നേടുക.arg_names = ['out']
	
	def execute_ദശാംശം_നേടുക(self, context):
		while True:
			text = input(str(context.symbol_table.get('out')))
			try: number = float(text); break
			except ValueError: print(f'\'{text}\' must be an [INT] or [FLOAT]. Try again!')

		return RuntimeResult().success(Number(number))
	execute_ദശാംശം_നേടുക.arg_names = ['out']
	
	def execute_മായ്ക്കുക(self, context):
		system('cls' if name == 'nt' else 'clear')
		return RuntimeResult().success(Nothing())
	execute_മായ്ക്കുക.arg_names = []

	def execute_ക്രമഫലം(self, context):
		res = RuntimeResult()

		value = context.symbol_table.get('value')
		if value.__hash__: return res.success(Number(hash(value)))

		return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_ILLEGALOP, f'Value \'{str(value)}\' is not hashable', context))
	execute_ക്രമഫലം.arg_names = ['value']

	def execute_തരം(self, context):
		return RuntimeResult().success(String(type(context.symbol_table.get('value')).__name__))
	execute_തരം.arg_names = ['value']
	
	def execute_മാറ്റുക(self, context):
		res = RuntimeResult()
		value = context.symbol_table.get('value')
		type_ = context.symbol_table.get('type')

		if not isinstance(type_, String): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'Second argument must be a [STRING]', context))
		if type_.value not in ('String', 'Int', 'Float', 'Bool', 'Array', 'List'): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'Can only convert items to [STRING], [INT], [FLOAT], [BOOL], [ARRAY], [LIST]', context))

		new_value = Nothing()
		if type_.value == 'String': new_value = String(str(value))
		elif hasattr(value, 'is_true') and type_.value == 'Bool': new_value = Bool(value.is_true())
		elif isinstance(value, (String, Number, Bool)):
			if type_.value == 'Int':
				try: new_value = Number(int(value.value))
				except Exception: return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, f'Could not convert \'{value.value}\' to an [INT]', context))
			elif type_.value == 'Float':
				try: new_value = Number(float(value.value))
				except Exception: return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, f'Could not convert \'{value.value}\' to a [FLOAT]', context))
		elif isinstance(value, (List, Array)):
			if type_.value == 'Array': new_value = Array(value.elements)
			elif type_.value == 'List': new_value = List(list(value.elements))
			else: return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'Can only convert [LIST] and [ARRAY] types to [LIST], [ARRAY], [STRING] or [BOOL]', context))
		elif isinstance(value, (BaseFunction, Dict)): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'Can only convert [FUNCTION], [FUNCTION]-derived, [OBJECT] and [DICTIONARY] types to [STRING] or [BOOL]', context))
		else: return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'Unknown value type', context))

		return res.success(new_value)
	execute_മാറ്റുക.arg_names = ['value', 'type']

	def execute_തിരുകുക(self, context):
		res = RuntimeResult()
		list_ = context.symbol_table.get('list')
		index = context.symbol_table.get('index')
		value = context.symbol_table.get('value')
		
		if not isinstance(list_, List): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'First argument must be a [LIST]', context))
		if not isinstance(index, Number): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'Second argument must be an [INT]', context))
		
		if isinstance(value, List):
			index_ = index.value
			for i in range(len(value.elements)):
				list_.elements.insert(index_, value.elements[i])
				index_ += 1
		else: list_.elements.insert(index.value, value)
		return res.success(Nothing())
	execute_തിരുകുക.arg_names = ['list', 'index', 'value']

	def execute_നീളം(self, context):
		res = RuntimeResult()
		value = context.symbol_table.get('value')

		if not isinstance(value, (List, Array, Dict, String)): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'Argument must be a [LIST], [ARRAY], [DICTIONARY] or [STRING]', context))

		if isinstance(value, (List, Array)): return res.success(Number(len(value.elements)))
		elif isinstance(value, Dict): return res.success(Number(len(value.dict)))
		elif isinstance(value, String): return res.success(Number(len(value.value)))
		return res.success(Nothing())
	execute_നീളം.arg_names = ['value']

	def execute_പിളർക്കുക(self, context):
		res = RuntimeResult()
		value = context.symbol_table.get('value')
		sep = context.symbol_table.get('separator')

		if not isinstance(value, String): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'First argument must be a [STRING]', context))
		if not isinstance(sep, String): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'Second argument must be a [STRING]', context))

		elements = []
		split_ = value.value.split(sep.value)
		for i in split_: elements.append(String(i))
		return res.success(List(elements))
	execute_പിളർക്കുക.arg_names = ['value', 'separator']

	def execute_ചേരുക(self, context):
		res = RuntimeResult()
		list_ = context.symbol_table.get('list')
		sep = context.symbol_table.get('separator')

		if not isinstance(list_, (List, Array)): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'First argument must be a [LIST] or [ARRAY]', context))
		if not isinstance(sep, String): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'Second argument must be a [STRING]', context))

		elements = [str(i) for i in list_.elements]

		joined = sep.value.join(elements)
		return res.success(String(joined))
	execute_ചേരുക.arg_names = ['list', 'separator']

	def execute_തിരികെവെക്കുക(self, context):
		res = RuntimeResult()
		value = context.symbol_table.get('value')
		sub_a = context.symbol_table.get('arg_a')
		sub_b = context.symbol_table.get('arg_b')

		if isinstance(value, String):
			if not isinstance(sub_a, String): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'Second argument must be a [STRING]', context))
			if not isinstance(sub_b, String): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'Third argument must be a [STRING]', context))
			return res.success(String(value.value.replace(sub_a.value, sub_b.value)))
		elif isinstance(value, List):
			if not isinstance(sub_a, Number): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'Second argument must be a [INT]', context))
			new_list = value.copy()
			new_list.elements[sub_a.value] = sub_b
			return res.success(new_list)

		return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'First argument must be a [STRING] or [LIST]', context))
	execute_തിരികെവെക്കുക.arg_names = ['value', 'arg_a', 'arg_b']

	def execute_നിർവഹിക്കുക(self, context):
		res = RuntimeResult()
		fn = context.symbol_table.get('filepath')

		if not isinstance(fn, String): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_INCORRECTTYPE, 'വാദം [സരണി] ആയിരിക്കണം', context))

		fn = fn.value
		if not path.isfile(fn): return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_IO, f'നിര \'{fn}\' നിലവിലില്ല', context))

		try:
			with open(fn, 'r', encoding='UTF-8') as f:
				script = f.read()
		except Exception as error: return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_IO, f'കുറിപ്പ് \'{fn}\' കയറ്റാൻ പറ്റീല:\n{str(error)}', context))
		
		_, error = run(fn, script)
		if error: return res.failure(RuntimeError(self.start_pos, self.end_pos, RTE_IO, f'കുറിപ്പ് \'{fn}\' നിർവ്വഹിക്കാൻ പറ്റീല:\n\n{error.as_string()}', context))
		return res.success(Nothing())
	execute_നിർവഹിക്കുക.arg_names = ['filepath']

	def copy(self):
		copy_ = BuiltInFunction(self.name)
		copy_.set_context(self.context)
		copy_.set_pos(self.start_pos, self.end_pos)
		return copy_

	def __repr__(self):
		return f'<നിർമ്മിച്ചിരിക്കുന്ന പ്രവർത്തനം <{self.name}>>'

	def __hash__(self):
		method_name = f'execute_{self.name}'
		method = getattr(self, method_name, self.no_visit_method)

		hash_value = hash(0)
		for i in method.arg_names: hash_value ^= hash(i)

		return hash(self.name) ^ hash_value

BuiltInFunction.show               = BuiltInFunction('കാണിക്കുക')
BuiltInFunction.show_error         = BuiltInFunction('പിശക്_കാണിക്കുക')
BuiltInFunction.get                = BuiltInFunction('നേടുക')
BuiltInFunction.get_int            = BuiltInFunction('പൂർണ്ണസംഖ്യ_നേടുക')
BuiltInFunction.get_float          = BuiltInFunction('ദശാംശം_നേടുക')
BuiltInFunction.clear_screen       = BuiltInFunction('മായ്ക്കുക')
BuiltInFunction.hash               = BuiltInFunction('ക്രമഫലം')
BuiltInFunction.type_of            = BuiltInFunction('തരം')
BuiltInFunction.convert            = BuiltInFunction('മാറ്റുക')
BuiltInFunction.insert             = BuiltInFunction('തിരുകുക')
BuiltInFunction.len                = BuiltInFunction('നീളം')
BuiltInFunction.split              = BuiltInFunction('പിളർക്കുക')
BuiltInFunction.join               = BuiltInFunction('ചേരുക')
BuiltInFunction.replace            = BuiltInFunction('തിരികെവെക്കുക')
BuiltInFunction.run                = BuiltInFunction('നിർവഹിക്കുക')

class Object(BaseFunction):
	def __init__(self, name, body_node, arg_names, internal_context=None):
		super().__init__(name)
		self.body_node = body_node
		self.arg_names = arg_names
		self.internal_context = internal_context

	def execute(self, args):
		res = RuntimeResult()
		interpreter = Interpreter()
		self.internal_context = self.generate_context()
		res.register(self.check_and_populate_args(self.arg_names, args, self.internal_context))
		if res.should_return(): return res

		res.register(interpreter.visit(self.body_node, self.internal_context))
		if res.should_return() and res.function_return_value == None: return res

		return res.success(self.copy())

	def retrieve(self, node):
		res = RuntimeResult()

		if self.internal_context:
			interpreter = Interpreter()
			value = res.register(interpreter.visit(node, self.internal_context))
			if res.should_return() and res.function_return_value == None: return res

			return_value = value
			return res.success(return_value)
		return res.failure(RuntimeError(node.start_pos, node.end_pos, RTE_ILLEGALOP, 'ആരംഭിക്കാത്ത വസ്തുവിൽ \'വീണ്ടെടുക്കുക\' പ്രവർത്തനം വിളിച്ചു', self.context))
	
	def copy(self):
		copy_ = Object(self.name, self.body_node, self.arg_names, self.internal_context)
		copy_.set_context(self.context)
		copy_.set_pos(self.start_pos, self.end_pos)
		return copy_
	
	def __repr__(self):
		return f'<വസ്തു <{self.name}>>'

	def __hash__(self):
		hash_value = hash(0)
		for i in self.arg_names: hash_value ^= hash(i)

		return hash(self.body_node) ^ hash(self.name) ^ hash(self.internal_context) ^ hash_value

# CONTEXT

class Context:
	def __init__(self, display_name, parent=None, parent_entry_pos=None):
		self.display_name = display_name
		self.parent = parent
		self.parent_entry_pos = parent_entry_pos
		self.symbol_table = None

# SYMBOL TABLE

class SymbolTable:
	def __init__(self, parent=None):
		self.symbols = {}
		self.parent = parent

	def get(self, name):
		value = self.symbols.get(name, None)

		if value == None and self.parent: return self.parent.get(name)
		return value
	
	def set(self, name, value):
		self.symbols[name] = value

	def remove(self, name):
		del self.symbols[name]

# INTERPRETER

class Interpreter:
	def visit(self, node, context):
		method_name= f'visit_{type(node).__name__}'
		method = getattr(self, method_name, self.no_visit_method)
		return method(node, context)

	def no_visit_method(self, node, context):
		raise Exception(f'No visit_{type(node).__name__} method defined!')

	def visit_NumberNode(self, node, context):
		return RuntimeResult().success(Number(node.token.value).set_context(context).set_pos(node.start_pos, node.end_pos))

	def visit_StringNode(self, node, context):
		return RuntimeResult().success(String(node.token.value).set_context(context).set_pos(node.start_pos, node.end_pos))

	def visit_ListNode(self, node, context):
		res = RuntimeResult()
		elements = []

		for element_node in node.element_nodes:
			elements.append(res.register(self.visit(element_node, context)))
			if res.should_return(): return res
		
		return res.success(List(elements).set_context(context).set_pos(node.start_pos, node.end_pos))

	def visit_ArrayNode(self, node, context):
		res = RuntimeResult()
		elements = []

		for element_node in node.element_nodes:
			elements.append(res.register(self.visit(element_node, context)))
			if res.should_return(): return res
		
		return res.success(Array(elements).set_context(context).set_pos(node.start_pos, node.end_pos))

	def visit_DictNode(self, node, context):
		res = RuntimeResult()
		main_dict = {}

		for key_node, value_node in node.pair_nodes:
			key = res.register(self.visit(key_node, context))
			if res.should_return(): return res

			value = res.register(self.visit(value_node, context))
			if res.should_return(): return res

			if not key.__hash__: return res.failure(RuntimeError(key.start_pos, key.end_pos, RTE_DICTKEY, f'താക്കോൽ \'{str(key)}\' ക്രമീകരിക്കാൻ പറ്റില്ല', context))
			main_dict[hash(key)] = (repr(key), value)
		
		return res.success(Dict(main_dict).set_context(context).set_pos(node.start_pos, node.end_pos))
		
	def visit_VarAccessNode(self, node, context):
		res = RuntimeResult()
		var_name = node.var_name_token.value
		value = context.symbol_table.get(var_name)

		if not value: return res.failure(RuntimeError(node.start_pos, node.end_pos, RTE_UNDEFINEDVAR, f'\'{var_name}\' നിർവചിച്ചിട്ടില്ല', context))

		value = value.copy().set_pos(node.start_pos, node.end_pos).set_context(context)
		return res.success(value)

	def visit_VarAssignNode(self, node, context):
		res = RuntimeResult()
		var_name = node.var_name_token.value
		value = res.register(self.visit(node.value_node, context))
		if res.should_return(): return res
		
		if node.is_global:
			parent_context = context
			while parent_context.parent: parent_context = parent_context.parent

			parent_context.symbol_table.set(var_name, value)
		else: context.symbol_table.set(var_name, value)
		return res.success(value)

	def visit_BinOpNode(self, node, context):
		res = RuntimeResult()
		left = res.register(self.visit(node.left_node, context))
		if res.should_return(): return res
		right = res.register(self.visit(node.right_node, context))
		if res.should_return(): return res

		if node.operator_token.type == TT_PLUS:
			result, error = left.added_to(right)
		elif node.operator_token.type == TT_MINUS:
			result, error = left.subbed_by(right)
		elif node.operator_token.type == TT_MUL:
			result, error = left.multed_by(right)
		elif node.operator_token.type == TT_DIV:
			result, error = left.dived_by(right)
		elif node.operator_token.type == TT_MOD:
			result, error = left.moded_by(right)
		elif node.operator_token.type == TT_POW:
			result, error = left.powed_by(right)
		
		elif node.operator_token.type == TT_IE:
			result, error = left.compare_eq(right)
		elif node.operator_token.type == TT_NE:
			result, error = left.compare_ne(right)
		elif node.operator_token.type == TT_LT:
			result, error = left.compare_lt(right)
		elif node.operator_token.type == TT_GT:
			result, error = left.compare_gt(right)
		elif node.operator_token.type == TT_LTE:
			result, error = left.compare_lte(right)
		elif node.operator_token.type == TT_GTE:
			result, error = left.compare_gte(right)
		elif node.operator_token.matches(TT_KEY, 'ഉം'):
			result, error = left.compare_and(right)
		elif node.operator_token.matches(TT_KEY, 'അല്ലെങ്കിൽ'):
			result, error = left.compare_or(right)

		elif node.operator_token.matches(TT_KEY, 'ൽ'):
			result, error = right.check_in(left)

		if error: return res.failure(error)
		return res.success(result.set_pos(node.start_pos, node.end_pos))

	def visit_UnaryOpNode(self, node, context):
		res = RuntimeResult()
		number = res.register(self.visit(node.node, context))
		if res.should_return(): return res

		error = None
		if node.operator_token.type == TT_MINUS: number, error = number.multed_by(Number(-1))
		elif node.operator_token.matches(TT_KEY, 'വിപരീതം'): number, error = number.invert()
		
		if error: return res.failure(error)
		return res.success(number.set_pos(node.start_pos, node.end_pos))

	def visit_IfNode(self, node, context):
		res = RuntimeResult()

		for condition, expression in node.cases:
			condition_value = res.register(self.visit(condition, context))
			if res.should_return(): return res

			if condition_value.is_true():
				expression_value = res.register(self.visit(expression, context))
				if res.should_return(): return res
				return res.success(Nothing() if node.should_return_null else expression_value)
		
		if node.else_case:
			expression = node.else_case
			else_value = res.register(self.visit(expression, context))
			if res.should_return(): return res
			return res.success(Nothing() if node.should_return_null else else_value)
		
		return res.success(Nothing())

	def visit_CountNode(self, node, context):
		res = RuntimeResult()
		elements = []

		start_value = res.register(self.visit(node.start_value_node, context))
		if res.should_return(): return res

		end_value = res.register(self.visit(node.end_value_node, context))
		if res.should_return(): return res

		if node.step_value_node:
			step_value = res.register(self.visit(node.step_value_node, context))
			if res.should_return(): return res
		else:
			step_value = Number(1)
		
		i = start_value.value
		if step_value.value > 0: condition = lambda: i < end_value.value
		else: condition = lambda: i > end_value.value

		while condition():
			context.symbol_table.set(node.var_name_token.value, Number(i))
			i += step_value.value

			value = res.register(self.visit(node.body_node, context))
			if res.should_return() and not res.loop_should_skip and not res.loop_should_stop: return res
			if res.loop_should_skip: continue
			if res.loop_should_stop: break
			elements.append(value)

		return res.success(Nothing() if node.should_return_null else Array(elements).set_context(context).set_pos(node.start_pos, node.end_pos))
	
	def visit_WhileNode(self, node, context):
		res = RuntimeResult()
		elements = []

		while True:
			condition = res.register(self.visit(node.condition_node, context))
			if res.should_return(): return res

			if not condition.is_true(): break

			value = res.register(self.visit(node.body_node, context))
			if res.should_return() and not res.loop_should_skip and not res.loop_should_stop: return res
			if res.loop_should_skip: continue
			if res.loop_should_stop: break
			elements.append(value)
		
		return res.success(Nothing() if node.should_return_null else Array(elements).set_context(context).set_pos(node.start_pos, node.end_pos))
	
	def visit_TryNode(self, node, context):
		res = RuntimeResult()
		value = res.register(self.visit(node.body_node, context))

		if res.error:
			error = res.error.error_type
			value = Nothing()
			res.reset()

			if len(node.catches) >= 1:
				for i in node.catches:
					if (i[0] == None):
						if i[1] != None: context.symbol_table.set(i[1].value, String(error))

						value = res.register(self.visit(i[2], context))
						if res.should_return(): return res
						break
					elif (i[0].value == error):
						if i[1] != None: context.symbol_table.set(i[1].value, String(error))

						value = res.register(self.visit(i[2], context))
						if res.should_return(): return res
						break

		return res.success(Nothing() if node.should_return_null else value.set_context(context).set_pos(node.start_pos, node.end_pos))

	def visit_FuncDefNode(self, node, context):
		res = RuntimeResult()

		func_name = node.var_name_token.value if node.var_name_token else None
		body_node = node.body_node
		arg_names = [arg_name.value for arg_name in node.arg_name_tokens]
		func_value = Function(func_name, body_node, arg_names, node.should_return_null).set_context(context).set_pos(node.start_pos, node.end_pos)

		if node.var_name_token: context.symbol_table.set(func_name, func_value)

		return res.success(func_value)

	def visit_ObjectDefNode(self, node, context):
		res = RuntimeResult()

		object_name = node.var_name_token.value
		body_node = node.body_node
		arg_names = [arg_name.value for arg_name in node.arg_name_tokens]
		object_value = Object(object_name, body_node, arg_names).set_context(context).set_pos(node.start_pos, node.end_pos)

		context.symbol_table.set(object_name, object_value)
		return res.success(object_value)

	def visit_CallNode(self, node, context):
		res = RuntimeResult()
		args = []

		value_to_call = res.register(self.visit(node.node_to_call, context))
		if res.should_return(): return res
		value_to_call = value_to_call.copy().set_pos(node.start_pos, node.end_pos)

		for arg_node in node.arg_nodes:
			args.append(res.register(self.visit(arg_node, context)))
			if res.should_return(): return res

		return_value = res.register(value_to_call.execute(args))

		if res.should_return(): return res
		return_value = return_value.copy().set_pos(node.start_pos, node.end_pos).set_context(context)
		return res.success(return_value)

	def visit_ObjectCallNode(self, node, context):
		res = RuntimeResult()

		object_ = res.register(self.visit(node.object_node, context))
		if res.should_return(): return res

		object_ = object_.copy().set_pos(node.start_pos, node.end_pos)

		return_value = res.register(object_.retrieve(node.node_to_call))
		if res.should_return(): return res
		return_value = return_value.copy().set_pos(node.start_pos, node.end_pos).set_context(context)
		return res.success(return_value)

	def visit_IncludeNode(self, node, context):
		res = RuntimeResult()

		filename = node.name_node.value
		plausible = [path.join(getcwd(), filename), filename, *[path.join(i, filename) for i in local_lib_path], path.join(LIB_PATH, filename)]

		filepath = None
		for i in plausible:
			if path.isfile(i):
				filepath = i
				break

		if filepath == None: return res.failure(RuntimeError(node.start_pos, node.end_pos, RTE_IO, f'നിര \'{filename}\' നിലവിലില്ല', context))

		if len(path.splitext(filepath)) > 1 and path.splitext(filepath)[1] == '.py':
			try:
				location, lib_name = path.abspath(filepath), path.splitext(path.basename(filepath))[0]
				spec = util.spec_from_file_location(lib_name, location)
				lib = util.module_from_spec(spec)
				spec.loader.exec_module(lib)
			except ImportError as error: return res.failure(RuntimeError(node.start_pos, node.end_pos, RTE_IO, f'കുറിപ്പ് \'{filename}\' കയറ്റാൻ പറ്റീല:\n\n{str(error).capitalize()}', context))

			try: object_ = res.register(lib.lib_Object().set_context(context).set_pos(node.start_pos, node.end_pos).execute())
			except AttributeError as error: return res.failure(RuntimeError(node.start_pos, node.end_pos, RTE_IO, f'കുറിപ്പ് \'{filename}\' നിർവ്വഹിക്കാൻ പറ്റീല:\n\n{str(error)}', context))

			if res.should_return(): return res
			name = node.nickname_node.value if node.nickname_node else object_.name

			fname = ''
			for i in name:
				if i not in ALPHANUM_UNDERSCORE: fname = f'{fname}_'
				else: fname = f'{fname}{i}'

			object_.name = fname
			context.symbol_table.set(fname, object_)
		else:
			try:
				with open(filepath, 'r', encoding='UTF-8') as f: script = f.read()
			except Exception as error: return res.failure(RuntimeError(node.start_pos, node.end_pos, RTE_IO, f'കുറിപ്പ് \'{filename}\' കയറ്റാൻ പറ്റീല:\n\n{str(error)}', context))

			tokens, error = Lexer(filename, script).compile_tokens()
			if error: return res.failure(RuntimeError(node.start_pos, node.end_pos, RTE_IO, f'കുറിപ്പ് \'{filename}\' നിർവ്വഹിക്കാൻ പറ്റീല:\n\n{error.as_string()}', context))

			ast = Parser(tokens).parse()
			if ast.error: return res.failure(RuntimeError(node.start_pos, node.end_pos, RTE_IO, f'കുറിപ്പ് \'{filename}\' നിർവ്വഹിക്കാൻ പറ്റീല:\n\n{ast.error.as_string()}', context))
		
			name = node.nickname_node.value if node.nickname_node else path.splitext(path.basename(node.name_node.value))[0]

			fname = ''
			for i in name:
				if i not in ALPHANUM_UNDERSCORE: fname = f'{fname}_'
				else: fname = f'{fname}{i}'

			object_ = res.register(Object(fname, ast.node, []).set_context(context).set_pos(node.start_pos, node.end_pos).execute([]))
			if res.should_return(): return res

			context.symbol_table.set(fname, object_)

		return res.success(Nothing())

	def visit_ReturnNode(self, node, context):
		res = RuntimeResult()

		if node.node_to_return:
			value = res.register(self.visit(node.node_to_return, context))
			if res.should_return(): return res
		else: value = Nothing()

		return res.success_return(value)
	
	def visit_SkipNode(self, node, context):
		return RuntimeResult().success_skip()

	def visit_StopNode(self, node, context):
		return RuntimeResult().success_stop()

# RUN

global_symbol_table = SymbolTable()

# Main variables
global_symbol_table.set('ഒന്നുമില്ല', Nothing.nothing)
global_symbol_table.set('സത്യം', Bool.true)
global_symbol_table.set('തെറ്റ്', Bool.false)
global_symbol_table.set('ഇസ്ർ_പതിപ്പ്', String.ezr_version)

# Main functions
global_symbol_table.set('കാണിക്കുക', BuiltInFunction.show)
global_symbol_table.set('പിശക്_കാണിക്കുക', BuiltInFunction.show_error)
global_symbol_table.set('നേടുക', BuiltInFunction.get)
global_symbol_table.set('പൂർണ്ണസംഖ്യ_നേടുക', BuiltInFunction.get_int)
global_symbol_table.set('ദശാംശം_നേടുക', BuiltInFunction.get_float)
global_symbol_table.set('മായ്ക്കുക', BuiltInFunction.clear_screen)
global_symbol_table.set('ക്രമഫലം', BuiltInFunction.hash)
global_symbol_table.set('തരം', BuiltInFunction.type_of)
global_symbol_table.set('മാറ്റുക', BuiltInFunction.convert)
global_symbol_table.set('തിരുകുക', BuiltInFunction.insert)
global_symbol_table.set('നീളം', BuiltInFunction.len)
global_symbol_table.set('പിളർക്കുക', BuiltInFunction.split)
global_symbol_table.set('ചേരുക', BuiltInFunction.join)
global_symbol_table.set('തിരികെവെക്കുക', BuiltInFunction.replace)
global_symbol_table.set('നിർവഹിക്കുക', BuiltInFunction.run)

local_lib_path = []

def run(fn, input_):
	global local_lib_path
	filepath = path.abspath(path.dirname(fn))
	local_lib_path.append(filepath)

	# Generate tokens
	lexer = Lexer(fn, input_)
	tokens, error = lexer.compile_tokens()
	if error: return None, error

	# Generate abstract syntax tree
	parser = Parser(tokens)
	ast = parser.parse()
	if ast.error: return None, ast.error

	# Run program
	interpreter = Interpreter()
	context = Context('<പ്രധാന സംഭവം>')
	context.symbol_table = global_symbol_table
	result = interpreter.visit(ast.node, context)

	return result.value, result.error