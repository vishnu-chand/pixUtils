import sys
sys.path.append('/home/hippo/SOFT/pixUtils')
sys.path.append('/home/hippo/SOFT/pixUtils/pixUtils')
from pixUtils import *


def getAwsId(ipPaths, endBy):
    data = list()
    for ipPath in ipPaths:
        if ipPath.endswith(endBy):
            with open(ipPath, 'r') as book:
                awsId = book.read()
                awsName = basename(ipPath).split('.aws')[1][:-1]
                data.append([ipPath, awsId, f".aws{awsName}"])
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
    if startIp:
        src, awsId, awsName = startIp
        try:
            exeIt(f'aws ec2 start-instances --instance-ids i-{awsId}', debug=False)
        except:
            pass
        ip = 'failToStart'
        for i in range(10):
            try:
                cmd, errCode, out, err = exeIt(f'aws ec2 describe-instances --instance-ids  i-{awsId}', returnOutput=True, debug=False)
                ip = json.loads(out)['Reservations'][0]['Instances'][0]['PublicIpAddress']
                break
            except Exception as exp:
                time.sleep(.3)
        dirop(src, mv=f'{dirname(src)}/{ip}{awsName}')
    elif endIp:
        src, awsId, awsName = endIp
        exeIt(f'aws ec2 stop-instances --instance-ids  i-{awsId} --force', debug=False)
        dirop(src, mv=f'{dirname(src)}/a{awsName}1')


opAws()
os.system('cd /home/hippo/SOFT/pycharm-professional/pycharm/bin;bash pycharm.sh')
