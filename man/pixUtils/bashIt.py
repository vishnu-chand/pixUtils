import os
import json
import subprocess as sp

bashEnv = os.getenv('bashEnv', 'export PATH=/home/ec2-user/miniconda3/bin:$PATH;eval "$(conda shell.bash hook)";conda activate gputf;')


def decodeCmd(cmd, sepBy):
    cmd = [cmd.strip() for cmd in cmd.split('\n')]
    cmd = [cmd for cmd in cmd if cmd and not cmd.startswith('#')]
    cmd = sepBy.join(cmd)
    return cmd


def exeIt(cmd, returnOutput=True, waitTillComplete=True, sepBy=' ', inData=None, debug=True, raiseOnException=True, skipExe=False):
    if returnOutput and not waitTillComplete:
        raise Exception("waitTillComplete is False, to get returnOutput set waitTillComplete to True")
    stdin = None if inData is None else sp.PIPE
    stdout, stderr = (None, None) if debug else (sp.DEVNULL, sp.DEVNULL)
    if returnOutput:
        stdout, stderr = sp.PIPE, sp.PIPE
    cmd = decodeCmd(cmd, sepBy)
    if skipExe:
        print(f"""
                    {bashEnv}{cmd}
                """)
    else:
        p1 = sp.Popen(f"{bashEnv}{cmd}", shell=True, stdin=stdin, stdout=stdout, stderr=stderr)
        errCode = 0
        out, err = '', ''
        if waitTillComplete:
            inData = None if inData is None else inData.encode()
            out, err = p1.communicate(inData)
            errCode = p1.poll()
            out = '' if out is None else out.decode().strip()
            err = '' if err is None else err.decode().strip()
        if raiseOnException and errCode:
            out = f"""
            cmd         : {cmd}
            returnCode  : {errCode}
            out1        : {out}
            out2        : {err}
            """
            raise Exception(f"subprocess failed: {out}")
        if debug:
            print(f"""
          _____________________________________________________________________________________
          cmd         : 
                        {cmd}
    
          returnCode  : {errCode}
          out1        : {out}
          out2        : {err}
          stderr      : {stderr}
          stdin       : {stdin}
          stdout      : {stdout}
    
          _____________________________________________________________________________________
          """)
        return cmd, errCode, out, err


def curlIt(url, data=None, method='POST', other='', timeout=60, debug=False, waitTillComplete=False):
    data = '' if data is None else f"-d '{json.dumps(data)}'"
    timeout = f'--max-time {timeout}' if timeout else ''
    curlCmd = f"curl -X {method.upper()} '{url}' {data} {timeout} {other}"
    return exeIt(cmd=curlCmd, returnOutput=waitTillComplete, waitTillComplete=waitTillComplete, sepBy='', debug=debug)
