import sys

sys.path.append('/home/hippo/SOFT/pixUtils')
sys.path.append('/home/hippo/SOFT/pixUtils/pixUtils')
from pixUtils import *


def data2bk():
    pycharm = '/home/hippo/.config/JetBrains/PyCharm2020.3'
    dirop(f'{pycharm}/options/jdk.table.xml', cp='/home/hippo/Desktop/pycharmBK/jdk.table.xml', rm=True)
    dirop(f'{pycharm}/options/sshConfigs.xml', cp='/home/hippo/Desktop/pycharmBK/sshConfigs.xml', rm=True)
    dirop(f'{pycharm}/options/webServers.xml', cp='/home/hippo/Desktop/pycharmBK/webServers.xml', rm=True)


def getIp():
    ips = glob('/home/hippo/Desktop/*.pycharmFix_s')
    if ips:
        exeIt('aws ec2 start-instances --instance-ids i-0a87813f15a6add4a', debug=False)
        while 1:
            try:
                cmd, errCode, out, err = exeIt('aws ec2 describe-instances --instance-ids i-0a87813f15a6add4a', returnOutput=True, debug=False)
                ip = json.loads(out)['Reservations'][0]['Instances'][0]['PublicIpAddress']
                break
            except Exception as exp:
                time.sleep(.3)
        dirop(ips[0], mv=f'/home/hippo/Desktop/{ip}.pycharmFix_')
    ips = glob('/home/hippo/Desktop/*.pycharmFix_e')
    if ips:
        exeIt('aws ec2 stop-instances --instance-ids i-0a87813f15a6add4a --force', debug=False)
        dirop(ips[0], mv=f'/home/hippo/Desktop/a.pycharmFix_')
        data2bk()
    ip = ''
    try:
        ipPath = glob('/home/hippo/Desktop/*.pycharmFix_')[0]
        res = basename(ipPath).replace('.pycharmFix_', '')
        if res != res.replace('-', '.'):
            res = res.replace('-', '.')
            dirop(ipPath, mv=f'/home/hippo/Desktop/{res}.pycharmFix_')
        a, b, c, d = res.split('.')
        ip = res
    except:
        pass
    print("15 getIp startPycharm ip: ", ip)
    return ip


def fixPycharm(ip, skeleton, sshId, webId):
    def removeRemote():
        pycharm = '/home/hippo/.config/JetBrains/PyCharm2020.3'
        dirop(f'{pycharm}/options/sshConfigs.xml', rm=True)
        dirop(f'{pycharm}/options/webServers.xml', rm=True)
        srcJdkTable = f'{pycharm}/options/jdk.table.xml'
        with open(srcJdkTable, 'r') as book:
            lines = book.read()
        jdkPat = '<jdk{data}/jdk>'
        oldjdks = re.findall(jdkPat.format(data='(.*)'), lines, re.DOTALL)[0]
        newjdks = re.findall(jdkPat.format(data='(.*?)'), lines, re.DOTALL)
        newjdks = [jdkPat.format(data=jdk) for jdk in newjdks if 'homePath value="sftp://ec2-user' not in jdk]
        lines = lines.replace(oldjdks, '')
        lines = lines.replace('<jdk/jdk>', '\n'.join(newjdks))
        with open(srcJdkTable, 'w') as book:
            book.write(lines)

    pycharm = '/home/hippo/.config/JetBrains/PyCharm2020.3'
    removeRemote()
    if ip == '':
        return
    jdkTable = f'{pycharm}/options/jdk.table.xml', f"""
    <jdk version="2">
      <name value="ec2-user" />
      <type value="Python SDK" />
      <version value="Python 3.6.12" />
      <homePath value="sftp://ec2-user@{ip}:22/home/ec2-user/miniconda3/envs/gputf/bin/python" />
      <roots>
        <classPath>
          <root type="composite">
            <root url="file://$USER_HOME$/.cache/JetBrains/PyCharm2020.3/remote_sources/{skeleton}/1553322517" type="simple" />
            <root url="file://$USER_HOME$/.cache/JetBrains/PyCharm2020.3/remote_sources/{skeleton}/1775388217" type="simple" />
            <root url="file://$USER_HOME$/.cache/JetBrains/PyCharm2020.3/python_stubs/{skeleton}" type="simple" />
            <root url="file://$APPLICATION_HOME_DIR$/plugins/python/helpers/python-skeletons" type="simple" />
            <root url="file://$APPLICATION_HOME_DIR$/plugins/python/helpers/typeshed/stdlib/3" type="simple" />
            <root url="file://$APPLICATION_HOME_DIR$/plugins/python/helpers/typeshed/stdlib/2and3" type="simple" />
            <root url="file://$APPLICATION_HOME_DIR$/plugins/python/helpers/typeshed/third_party/3" type="simple" />
            <root url="file://$APPLICATION_HOME_DIR$/plugins/python/helpers/typeshed/third_party/2and3" type="simple" />
          </root>
        </classPath>
        <sourcePath>
          <root type="composite" />
        </sourcePath>
      </roots>
      <additional INTERPRETER_PATH="/home/ec2-user/miniconda3/envs/gputf/bin/python" HELPERS_PATH="/home/ec2-user/.pycharm_helpers" INITIALIZED="false" VALID="true" RUN_AS_ROOT_VIA_SUDO="false" SKELETONS_PATH="" VERSION="" WEB_SERVER_CONFIG_ID="{webId}" WEB_SERVER_CONFIG_NAME="ec2-user" WEB_SERVER_CREDENTIALS_ID="sftp://ec2-user@{ip}:22">
        <PathMappingSettings>
          <option name="pathMappings">
            <list>
              <mapping local-root="$USER_HOME$/.cache/JetBrains/PyCharm2020.3/remote_sources/{skeleton}/1553322517" remote-root="/home/ec2-user/miniconda3/envs/gputf/lib/python3.6" />
              <mapping local-root="$USER_HOME$/.cache/JetBrains/PyCharm2020.3/remote_sources/{skeleton}/1775388217" remote-root="/home/ec2-user/miniconda3/envs/gputf/lib/python3.6/site-packages" />
            </list>
          </option>
        </PathMappingSettings>
      </additional>
    </jdk>
    """
    sshConfigs = f'{pycharm}/options/sshConfigs.xml', f"""
<application>
  <component name="SshConfigs">
    <configs>
      <sshConfig host="{ip}" id="{sshId}" keyPath="$USER_HOME$/awsBridge/.sshpy/vishnu.pem" port="22" customName="ec2-user" nameFormat="CUSTOM" username="ec2-user">
        <option name="customName" value="ec2-user" />
      </sshConfig>
    </configs>
  </component>
</application>
"""
    webServers = f'{pycharm}/options/webServers.xml', f"""
<application>
  <component name="WebServers">
    <option name="servers">
      <webServer id="{webId}" name="ec2-user">
        <fileTransfer accessType="SFTP" host="{ip}" port="22" sshConfigId="{sshId}" sshConfig="ec2-user" keyPair="true" />
      </webServer>
    </option>
  </component>
</application>
"""

    with open(jdkTable[0], 'r') as book:
        data = book.read().split('\n')
    data = '\n'.join(data[:-2]) + jdkTable[1] + '\n'.join(data[-2:])

    with open(dirop(jdkTable[0]), 'w') as book:
        book.write(data)

    with open(dirop(sshConfigs[0]), 'w') as book:
        book.write(sshConfigs[1])

    with open(dirop(webServers[0]), 'w') as book:
        book.write(webServers[1])


def getFixDatas(ip):
    with open(f'/home/hippo/Desktop/pycharmBK/webServers.xml', 'r') as book:
        data = book.read()  # .split('<fileTransfer accessType="SFTP"')[1].split('/>')[0]
    res = dict()
    res['ip'] = ip
    res['sshId'] = data.split('sshConfigId="')[1].split('" ')[0]
    res['webId'] = data.split('webServer id="')[1].split('" ')[0]

    with open(f'/home/hippo/Desktop/pycharmBK/jdk.table.xml', 'r') as book:
        res['skeleton'] = book.read().split('file://$USER_HOME$/.cache/JetBrains/PyCharm2020.3/remote_sources/')[1].split('/')[0]

    return res


# def fixIdea(ip):
#     for i in rglob('/home/hippo/awsBridge/virtualBGv4/HRNet-Semantic-Segmentation/.idea', '*.*'):
#         try:
#             with open(i, 'rb') as book:
#                 ilines = book.read().decode()
#             if ip in ''.join(ilines.split()):
#                 print(f"_______________________________{i}_______________________________")
#                 # with open(f'/home/hippo/Downloads/deleteMe/temp/{basename(i)}', 'w') as book:
#                 #     book.write(ilines)
#         except Exception as exp:
#             pass
#             # print('failed', i)
#             # print(exp)


# def test2(ip):
#     srcPaths = []
#     srcPaths.append('/home/hippo/.config/JetBrains')
#     srcPaths.append('/home/hippo/.cache/JetBrains')
#     srcPaths.append('/home/hippo/.local/share/JetBrains')
#     for srcPath in srcPaths:
#         for i in rglob(srcPath, '*.*'):
#             try:
#                 with open(i, 'rb') as book:
#                     ilines = book.read().decode()
#                 if ip in ''.join(ilines.split()):
#                     print(f"_______________________________{i}_______________________________")
#                     with open(f'/home/hippo/Downloads/deleteMe/pycharamFix/temp/{basename(i)}', 'w') as book:
#                         book.write(ilines)
#             except Exception as exp:
#                 pass
#                 # print('failed', i)
#                 # print(exp)


def bk2data(ip):
    pycharm = '/home/hippo/.config/JetBrains/PyCharm2020.3'
    books = list()
    books.append([f'{pycharm}/options/jdk.table.xml', '/home/hippo/Desktop/pycharmBK/jdk.table.xml'])
    books.append([f'{pycharm}/options/sshConfigs.xml', '/home/hippo/Desktop/pycharmBK/sshConfigs.xml'])
    books.append([f'{pycharm}/options/webServers.xml', '/home/hippo/Desktop/pycharmBK/webServers.xml'])
    srcIp = None
    for des, src in books:
        dirop(src, cp=des, rm=True)
        with open(src, 'r') as book:
            lines = book.read()
        if srcIp is None:
            srcIp = lines.split('sftp://ec2-user@')[1].split(':')[0]
        lines = lines.replace(srcIp, ip)
        with open(des, 'w') as book:
            book.write(lines)


ip = getIp()
# datas = getFixDatas(ip)
# fixPycharm(**datas)
# bk2data(datas['ip'])
# fixPycharm(ip='', skeleton='940137813', sshId="3dba16ba-1924-487d-9832-ca787097f435", webId="6de77690-ce19-4bc6-89b3-e660d4a0b2ff")
os.system('cd /home/hippo/SOFT/pycharm-professional-2020.2/pycharm-2020.2/bin;bash pycharm.sh')
