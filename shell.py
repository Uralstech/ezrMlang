import ezrM
from sys import argv
from requests import get, ConnectionError
GITHUB = 'https://github.com/Uralstech/ezrMlang/releases'

def check_version():
	try:
		ov_text = get('https://pastebin.com/raw/Qz8bKPpN').text
		ov, cv = ov_text.split('.'), ezrM.VERSION.split('.')
		for i, v in enumerate(cv):
			if ov[i] > v:
				if i == 0: print(f'UPDATE AVAILABLE: v{ov_text} [MAJOR UPDATE] -> {GITHUB}'); return
				elif i == 1: print(f'UPDATE AVAILABLE: v{ov_text} [Feature update] -> {GITHUB}'); return
				elif i == 2: print(f'UPDATE AVAILABLE: v{ov_text} [Function update] -> {GITHUB}'); return
				elif i == 3: print(f'UPDATE AVAILABLE: v{ov_text} [Library update] -> {GITHUB}'); return
				elif i == 4: print(f'UPDATE AVAILABLE: v{ov_text} [Patch] -> {GITHUB}'); return
			elif ov[i] < v: return
	except ConnectionError: print('Warning: Could not check for latest ezrM version')

def main():
	print(f'ezr Malayalam Shell v{ezrM.VERSION} ({ezrM.VERSION_DATE}) - Ctrl+C to exit')
	check_version()

	first_command = None
	if len(argv) > 1:
		path = argv[1].replace('\\', '//')
		first_command = f'നിർവഹിക്കുക(\'{path}\')'

	while True:
		try:
			if first_command == None:
				input_ = input('>>> ')
				if input_.strip() == '': continue
			else:
				print(f'>>> {first_command}')
				input_ = first_command
				first_command = None

			result, error = ezrM.run('__main__', input_)

			if error: print(error.as_string())
			elif result:
				if len(result.elements) == 1: print(repr(result.elements[0]))
				else: print(repr(result))
		except KeyboardInterrupt: break
		except EOFError: break

if __name__ == '__main__':
	main()