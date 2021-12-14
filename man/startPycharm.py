import os
import time
import json
import shutil
from glob import glob
import subprocess as sp
from os.path import basename, dirname

bashEnv = os.getenv('bashEnv', 'export PATH=/home/ec2-user/miniconda3/bin:$PATH;eval "$(conda shell.bash hook)";conda activate gputf;')


def renameFile(ips, oldPath, newPath):
    ips = sorted(ips, key=lambda x: os.path.splitext(x)[1].split('_')[0].replace('.aws', ''))
    datas = []
    for path in ips:
        with open(path, 'r') as book:
            data = book.read()
        datas.append(data)
        os.remove(path)
    for data, path in zip(datas, ips):
        mv = path
        if path == oldPath:
            mv = newPath
        p, e = os.path.splitext(mv)
        if not oldPath.endswith('e') and path != oldPath:  # deactivate other running instance
            e = f"{e.split('_')[0]}_1"
        d, f = dirname(p), basename(p)
        mv = f"{d}/{f.replace('-', '.')}{e}"
        with open(mv, 'w') as book:
            book.write(data)
        time.sleep(.15)


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
      stderr      : {stderr}
      stdin       : {stdin}
      stdout      : {stdout}

      _____________________________________________________________________________________
      """)
    return cmd, errCode, out, err


def getAwsId(ipPaths, endBy):
    data = list()
    for ipPath in ipPaths:
        if ipPath.endswith(endBy):
            with open(ipPath, 'r') as book:
                awsId = book.read().strip().split('\n')[0]
                awsName = basename(ipPath).split('.aws')[1].split('_')[0]
                data.append([ipPath, awsId, f".aws{awsName}_"])
    if data:
        if len(data) != 1:
            for d in data:
                print(d)
            raise Exception(f"more than one active instance disable one {data}")
        data = data[0]
    return data


def opAws():
    ips = glob('/home/hippo/Desktop/*.aws*')
    startIp = getAwsId(ips, endBy='s')
    endIp = getAwsId(ips, endBy='e')
    changeIp = getAwsId(ips, endBy='f')
    if startIp:
        src, awsId, awsName = startIp
        try:
            exeIt(f'aws ec2 start-instances --instance-ids {awsId}', debug=False)
        except:
            pass
        ip = 'failToStart'
        for i in range(10):
            try:
                cmd, errCode, out, err = exeIt(f'aws ec2 describe-instances --instance-ids  {awsId}', returnOutput=True, debug=False)
                ip = json.loads(out)['Reservations'][0]['Instances'][0]['PublicIpAddress']
                break
            except Exception as exp:
                time.sleep(.3)
        renameFile(ips, src, newPath=f'{dirname(src)}/{ip}{awsName}')
    elif endIp:
        src, awsId, awsName = endIp
        exeIt(f'aws ec2 stop-instances --instance-ids  {awsId} --force', debug=False)
        renameFile(ips, src, newPath=f'{dirname(src)}/a{awsName}1')
    elif changeIp:
        src, awsId, awsName = changeIp
        ip = basename(src).split('.aws')[0].replace('_', '.').replace('-', '.')
        renameFile(ips, src, newPath=f'{dirname(src)}/{ip}{awsName}')


opAws()
os.system('cd /home/hippo/SOFT/pycharm-professional/pycharm/bin;bash pycharm.sh')
