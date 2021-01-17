from pixUtils import *

delimiter = f"# =============================================================================================================================================sdfgver%$^#$^&*%^fd321"
autoCommentDelimiter = '""" auto comment starts', '""" auto comment ends'


def translator(lines, opType, cellType):
    if opType == 'ipynb2py':
        if cellType == 'code':
            pyLine = []
            for oline in lines:
                line = replaces(oline, ' ', '', '\n', '')
                nonPy = '%' in line[:2] or '!' in line[:2]
                if nonPy:
                    oline = f'# {oline}'
                pyLine.append(oline)
        else:
            markdown = f"'''\n{''.join(lines)}\n'''"
            pyLine = [markdown]
    else:
        if cellType == 'code':
            pyLine = []
            for oline in lines:
                line = replaces(oline, ' ', '', '\n', '')
                nonPy = '#%' in line[:3] or '#!' in line[:3]
                if nonPy:
                    oline = oline.lstrip().lstrip('#')[1:]
                pyLine.append(oline)
        else:
            pyLine = ['\n'.join(lines[1:-1])]
    return pyLine


def _py2ipynb(pyFile, desPath):
    with open(pyFile, 'r') as book:
        cells = book.read().split(delimiter)
    jcells = []
    for lines in cells[1:]:
        lines = lines.split('\n')
        cellType, lines = lines[0], '\n'.join(lines[1:])
        lines = lines.strip('\n').split('\n')
        lines = translator(lines, 'py2ipynb', cellType)
        lines = '\n'.join(lines)
        cell = dict(cell_type=cellType, metadata={}, source=lines)
        if cellType == 'code':
            cell = dict(cell_type=cellType, metadata={}, source=lines, execution_count=None, outputs=[])
        jcells.append(cell)
    if desPath is None:
        desPath = pyFile.replace('.py', 'sss.ipynb')
    json.dump(dict(cells=jcells, nbformat=4, nbformat_minor=0, metadata={}), open(desPath, 'w'))


def _ipynb2py(ipynb, desPath):
    jdata = json.load(open(ipynb))
    if desPath is None:
        desPath = ipynb.replace('.ipynb', '.py')
    with open(desPath, 'w') as book:
        for cell in jdata['cells']:
            cellType, lines = cell['cell_type'], cell['source']
            lines = translator(lines, 'ipynb2py', cellType)
            msgs = list(['\n'])
            msgs.append(f"{delimiter}{cellType}")
            msgs.append(''.join(lines))
            msgs = '\n'.join(msgs)
            book.write(msgs)
            # print(msgs)


def ipynb2py(path, desPath=None):
    if path.endswith('.ipynb'):
        _ipynb2py(path, desPath)
    else:
        _py2ipynb(path, desPath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--path", default=None, type=type(''), help=" eg: ")
    parser.add_argument("-d", "--desPath", default=None, type=type(''), help=" eg: ")
    userArgs = parser.parse_args()
    if userArgs.path is None:
        for path in glob('src/*.*'):
            desName = basename(path).replace('.py', '.ipynb')
            if path.endswith('.ipynb'):
                desName = basename(path).replace('.ipynb', '.py')
            ipynb2py(path, dirop(f'des/{desName}'))
    else:
        ipynb2py(userArgs.path, userArgs.desPath)
