import os
import json
import traceback
import subprocess as sp
from os.path import basename

# bashEnv = os.getenv('bashEnv', 'export PATH=/home/ec2-user/miniconda3/bin:$PATH;eval "$(conda shell.bash hook)";conda activate gputf;')


bashEnv = os.getenv('bashEnv', '')


def decodeCmd(cmd, sepBy):
    cmd = [cmd.strip() for cmd in cmd.split('\n')]
    cmd = [cmd for cmd in cmd if cmd and not cmd.startswith('#')]
    cmd = sepBy.join(cmd)
    return cmd


def getTraceBack(searchPys=None, tracebackData=None):
    errorTraceBooks = [basename(p) for p in searchPys or []]
    otrace = tracebackData or traceback.format_exc()
    trace = otrace.strip().split('\n')
    msg = trace[-1]
    done = False
    traces = [line.strip() for line in trace if line.strip().startswith('File "')]
    errLine = ''
    for line in traces[::-1]:
        if done:
            break
        meta = line.split(',')
        pyName = basename(meta[0].split(' ')[1].replace('"', ''))
        for book in errorTraceBooks:
            if book == pyName:
                done = True
                msg = f"{msg}, {' '.join(meta[1:])}. {meta[0]}"
                errLine = line
                break
    traces = '\n'.join(traces)
    traces = f"""
{msg}    


{otrace}


{traces}


{errLine}
"""
    return msg, traces


def exeIt(cmd, returnOutput=True, waitTillComplete=True, sepBy=' ', inData=None, debug=True, raiseOnException=True, skipExe=False):
    if returnOutput and not waitTillComplete:
        raise Exception("waitTillComplete is False, to get returnOutput set waitTillComplete to True")
    stdin = None if inData is None else sp.PIPE
    stdout, stderr = (None, None) if debug else (sp.DEVNULL, sp.DEVNULL)
    if returnOutput:
        stdout, stderr = sp.PIPE, sp.PIPE
    cmd = f"{bashEnv}{decodeCmd(cmd, sepBy)}"
    errCode, out, err = 0, 'no output', 'no output'
    if skipExe:
        print(f"""
        skipping execution of 
                    {cmd}
                """)
    elif not returnOutput and stdin is None and waitTillComplete:
        errCode = os.system(cmd)
    else:
        p1 = sp.Popen(cmd, shell=True, stdin=stdin, stdout=stdout, stderr=stderr)
        errCode, out, err = 0, '', ''
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
        out1[out]   : {out}
        out2[err]   : {err}
        """
        if returnOutput:
            _, out = getTraceBack(tracebackData=out)
        raise Exception(f"subprocess failed: {out}")
    if debug and not skipExe:
        print(f"""
      _____________________________________________________________________________________
      cmd         : 
                    {cmd}

      returnCode  : {errCode}
      out1[out]   : {out}
      out2[err]   : {err}
      stderr      : {stderr}
      stdin       : {stdin}
      stdout      : {stdout}

      _____________________________________________________________________________________
      """)
    return cmd, errCode, out, err


def curlIt(url, data=None, method='POST', other='', timeout=60, debug=False, waitTillComplete=False, skipExe=False):
    data = '' if data is None else f"-d '{json.dumps(data)}'"
    timeout = f'--max-time {timeout}' if timeout else ''
    curlCmd = f"curl -X {method.upper()} '{url}' {data} {timeout} {other}"
    return exeIt(cmd=curlCmd, returnOutput=waitTillComplete, waitTillComplete=waitTillComplete, sepBy='', debug=debug, skipExe=skipExe)
