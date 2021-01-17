import os
import json
import subprocess as sp

bashEnv = os.getenv('bashEnv', '')


def decodeCmd(cmd, sepBy):
    cmd = [cmd.strip() for cmd in cmd.split('\n')]
    cmd = [cmd for cmd in cmd if cmd and not cmd.startswith('#')]
    cmd = sepBy.join(cmd)
    return cmd


def exeIt(cmd, returnOutput=True, waitTillComplete=True, sepBy=' ', inData=None, debug=False, enableException=True):
    if returnOutput and not waitTillComplete:
        raise Exception("waitTillComplete is False, to get returnOutput set waitTillComplete to True")
    stdin = None if inData is None else sp.PIPE
    stdout, stderr = (None, None) if debug else (sp.DEVNULL, sp.DEVNULL)
    if returnOutput:
        stdout, stderr = sp.PIPE, sp.PIPE
    cmd = decodeCmd(cmd, sepBy)
    p1 = sp.Popen(f"{bashEnv}{cmd}", shell=True, stdin=stdin, stdout=stdout, stderr=stderr)
    errCode = 0
    out, err = '', ''
    if waitTillComplete:
        inData = None if inData is None else inData.encode()
        out, err = p1.communicate(inData)
        errCode = p1.poll()
        out = '' if out is None else out.decode().strip()
        err = '' if err is None else err.decode().strip()
    if enableException and errCode:
        out = f"""
        cmd         : {cmd}
        returnCode  : {errCode}
        pass        : {out}
        fail        : {err}
        """
        raise Exception(f"subprocess failed: {out}")
    if debug:
        print(f"""
      _____________________________________________________________________________________
      cmd         : 
                    {cmd}

      returnCode  : {errCode}
      pass        : {out}
      fail        : {err}
      _____________________________________________________________________________________
      """)
    return cmd, errCode, out, err


def curlIt(data=None, host='', port='', call='', url='', method='POST', other='', timeout=60, debug=False, waitTillComplete=False):
    if not url:
        url = f'{host}:{port}'
    if call:
        url = f'{url}/{call}'
    data = '' if data is None else f"-d '{json.dumps(data)}'"
    timeout = f'--max-time {timeout}' if timeout else ''
    curlCmd = f"curl -X {method.upper()} '{url}' {data} {timeout} {other}"
    return exeIt(cmd=curlCmd, returnOutput=waitTillComplete, waitTillComplete=waitTillComplete, sepBy='', debug=debug)
