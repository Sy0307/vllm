"""pb (push-beyond) 2P1D lifecycle on prod-09 (P + store master + proxy) and prod-24 (D + store client).

usage (on the node):  pb.py start-p RUN [k=v ...] | start-d RUN [k=v ...] | start-proxy RUN [k=v ...]
                      | start-dyn RUN p|d [k=v ...] | stop RUN p|d | status RUN p|d
start-dyn runs the Dynamo routing layer instead of proxy.py (engines started with DYN=1 expose gRPC):
p = etcd + one prefill sidecar per P instance + the Dynamo frontend on the proxy port; d = the decode sidecar.
Children are recorded with pid/start_ticks; stop kills only those process groups, after checking that
every member writes to this run's log files. Knobs (k=v) override the env passed to serve.sh.
"""
import datetime
import json
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

W = Path('/gpfs/mszn/workspace/dsv41-di-vllm-20261021')
PB = W / 'pb'
RT = PB / 'runtime'
V = W / 'rebase/f4edf9f0-20261023/.venv'
MC = PB / 'mcpath/mooncake'
# Node assignment: runtime/nodes.env next to this file (default prod-09 P + prod-24 D).
_NODES = dict(ln.split('=', 1) for ln in (Path(__file__).resolve().parent / 'nodes.env').read_text().split('\n')
              if '=' in ln and not ln.startswith('#'))
P_HOST, D_HOST = _NODES['P_IP'], _NODES['D_IP']
HOSTS = {'p': _NODES['P_HOSTNAME'], 'd': _NODES['D_HOSTNAME'],
         'dp': _NODES['P_HOSTNAME'],  # dp: a D-role validation server on the P host (no P running)
         'p24': _NODES['D_HOSTNAME']}  # p24: a P-only microbench on the D host (NODE=p24; no D running)
NODE_IP = {'p': P_HOST, 'dp': P_HOST, 'd': D_HOST, 'p24': D_HOST}
MC_PORT, MC_METRICS = 50061, 9063
NICS = _NODES['IB_NICS']
P_ROWS = [('prefill0', '0,1,2,3', '0 0 0 0', 8301, 14645, 30850, 5920),
          ('prefill1', '4,5,6,7', '1 1 1 1', 8302, 14745, 30950, 5930)]
D_ROW = ('decode', '0,1,2,3,4,5,6,7', '0 0 0 0 1 1 1 1', 8311, 14845, 31050, 6010)
PROXY_PORT = 8300
# Dynamo (user-approved isolated install, 2026-10-25): no-deps wheels in DYN_SITE; etcd copied from canlin's bin.
DYN_SITE = PB / 'dynamo-site-1.6.0.dev20261010'
ETCD = PB / 'bin/etcd'
ETCD_PORT, ETCD_PEER = 23790, 23800
GRPC = {'prefill0': 9301, 'prefill1': 9302, 'decode': 9311}


def now():
    return datetime.datetime.now().astimezone().isoformat()


def save(path, obj):
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def procstat(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()


def gpu_rows():
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                   text=True, timeout=30).split()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                   text=True, timeout=30).split()
    return [int(x) for x in used], apps


def assert_free():
    try:
        used, apps = gpu_rows()
    except subprocess.TimeoutExpired as exc:
        raise AssertionError('nvidia-smi timeout') from exc
    # A few MiB can stay on a GPU after its MPS daemon quits (driver residue, no process).
    assert len(used) == 8 and max(used) <= 64 and not apps, (used, apps)


def port_free(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('0.0.0.0', port))
            return True
        except OSError:
            return False


def mem_available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024**2


def guard(node, ports, min_avail_gib):
    assert socket.gethostname() == HOSTS[node], socket.gethostname()
    assert W.stat().st_uid == os.getuid()
    assert_free()
    busy = [p for p in ports if not port_free(p)]
    assert not busy, f'ports busy {busy}'
    avail = mem_available_gib()
    assert avail > min_avail_gib, f'MemAvailable {avail:.0f} GiB <= {min_avail_gib}'


class Run:
    def __init__(self, run, node):
        self.out = PB / 'runs' / run
        self.out.mkdir(parents=True, exist_ok=True)
        self.node = node
        self.path = self.out / f'children-{node}.json'
        self.children = json.loads(self.path.read_text()) if self.path.exists() else {}

    def spawn(self, name, command, env):
        env = {**env, 'PB_RUN': self.out.name}
        log = self.out / f'{name}.log'
        with log.open('a') as handle:
            p = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True, cwd='/tmp')
        self.children[name] = {'pid': p.pid, 'start_ticks': procstat(p.pid)[19], 'command': command,
                               'log': str(log), 'started': now()}
        save(self.path, self.children)
        return p.pid

    def alive(self, name):
        c = self.children[name]
        try:
            st = procstat(c['pid'])
            return st[0] != 'Z' and st[19] == c['start_ticks']
        except FileNotFoundError:
            return False


def base_env():
    env = dict(os.environ)
    for k in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
        env.pop(k, None)
    env['NO_PROXY'] = env['no_proxy'] = f'127.0.0.1,localhost,{P_HOST},{D_HOST}'
    return env


def knobs(args):
    """k=v pairs, or @name to read runtime/knobs/name.env (k=v lines; {now+N} -> epoch seconds)."""
    out = {}
    for a in args:
        if a.startswith('@'):
            for line in (RT / 'knobs' / f'{a[1:]}.env').read_text().splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    v = re.sub(r'\{now\+([0-9.]+)\}', lambda m: str(int(time.time() + float(m.group(1)))), v)
                    out[k] = v
        else:
            k, v = a.split('=', 1)
            out[k] = v
    return out


def health(url, timeout=3):
    try:
        with urllib.request.urlopen(url + '/health', timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def start_p(run, args):
    k = knobs(args)
    node = k.pop('NODE', 'p')
    host_ip = NODE_IP[node]
    ports = [PROXY_PORT, MC_PORT, MC_METRICS] + [r[3] for r in P_ROWS] + [r[4] for r in P_ROWS] \
        + [r[6] + i for r in P_ROWS for i in range(4)]
    guard(node, ports, float(k.pop('MIN_AVAIL_GIB', 1300)))
    r = Run(run, node)
    seg = k.pop('MC_SEG', '104GB')
    mc_cfg = r.out / 'mooncake_p.json'
    save(mc_cfg, {'mode': 'embedded', 'metadata_server': 'P2PHANDSHAKE',
                  'master_server_address': f'{host_ip}:{MC_PORT}', 'global_segment_size': seg,
                  'local_buffer_size': '4GB', 'protocol': 'rdma', 'device_name': NICS, 'enable_offload': False})
    env = base_env()
    r.spawn('mooncake_master', [str(MC / 'mooncake_master'), f'--rpc_port={MC_PORT}',
                                f'--metrics_port={MC_METRICS}',
                                f'--eviction_high_watermark_ratio={k.pop("MC_HIGH_WM", "0.95")}',
                                '--default_kv_lease_ttl=10000', '--rpc_thread_num=16'], env)
    time.sleep(5)
    assert r.alive('mooncake_master'), 'mooncake_master exited'
    dyn = k.pop('DYN', '0') == '1'
    for i, (name, gpus, numa, port, rpc, master, nixl) in enumerate(P_ROWS):
        # Knobs prefixed P0_/P1_ apply to that instance only (heterogeneous prefill instances).
        common = {kk: v for kk, v in k.items() if not kk.startswith(('P0_', 'P1_'))}
        own = {kk[3:]: v for kk, v in k.items() if kk.startswith(f'P{i}_')}
        e = {**env, 'ROLE': 'prefill', 'GPUS': gpus, 'NUMA': numa, 'PORT': str(port), 'RPC_PORT': str(rpc),
             'DP_MASTER_PORT': str(master), 'NIXL_SIDE_CHANNEL_PORT': str(nixl), 'NIXL_HOST': host_ip,
             'MC_CONFIG': str(mc_cfg), 'DP': '4', **common, **own}
        if dyn:
            e['GRPC_PORT'] = str(GRPC[name])
        r.spawn(name, ['bash', str(RT / 'serve.sh')], e)
    save(r.out / 'start-p.json', {'time': now(), 'knobs': k, 'mc_seg': seg})
    print('STARTED-P', json.dumps({n: c['pid'] for n, c in r.children.items()}))


def start_proxy(run, args):
    k = knobs(args)
    r = Run(run, 'p')
    assert socket.gethostname() == HOSTS['p'] and port_free(PROXY_PORT)
    cmd = [str(V / 'bin/python'), str(RT / 'proxy.py'), '--port', str(PROXY_PORT),
           '--host', k.pop('PROXY_HOST', '127.0.0.1'),
           '--prefill', f'http://{P_HOST}:{P_ROWS[0][3]}', '--prefill', f'http://{P_HOST}:{P_ROWS[1][3]}',
           '--decode', f'http://{D_HOST}:{D_ROW[3]}', '--routing', 'kv', '--disable-early-token']
    if k.get('COST_ROUTING', '1') == '1':
        cmd.append('--prefill-cost-routing')
    if k.get('SPILL_MS'):
        cmd += ['--spill-ms', k['SPILL_MS']]
    if k.get('D_AFFINITY_SKEW'):
        cmd += ['--d-affinity-skew', k['D_AFFINITY_SKEW']]
    if k.get('MESSAGE_ROUTING', '0') == '1':
        cmd.append('--message-routing')
    if k.get('D_CTX_WEIGHT'):
        cmd += ['--d-ctx-weight', k['D_CTX_WEIGHT']]
    if k.get('CQ_CAP_MS'):
        cmd += ['--cq-cap-ms', k['CQ_CAP_MS'], '--cq-spill-ms', k.get('CQ_SPILL_MS', '100'),
                '--cq-rate', k.get('CQ_RATE', '20'), '--cq-req-cap-ms', k.get('CQ_REQ_CAP_MS', '150')]
        if k.get('CQ_FAST_WORK'):
            cmd += ['--cq-fast-work', k['CQ_FAST_WORK']]
        if k.get('CQ_P_PRIORITY', '0') == '1':
            cmd.append('--cq-p-priority')
    if k.get('SJF_RATE'):
        cmd += ['--sjf-rate', k['SJF_RATE']]
    env = {**base_env(), 'PYTHONPATH': str(RT)}
    r.spawn('proxy', cmd, env)
    print('STARTED-PROXY', r.children['proxy']['pid'], ' '.join(cmd))


def mc_client_cmd(seg):
    return [str(MC / 'mooncake_client'), f'-master_server_address={P_HOST}:{MC_PORT}', '-metadata_server=P2PHANDSHAKE',
            '-protocol=rdma', f'-device_names={NICS}', f'-host={D_HOST}', f'-global_segment_size={seg}',
            '-local_buffer_size=1GB', '-port=50062']


def restart_mc_client(run):
    """Respawn D's Mooncake store client after it died. It only donates D-host DRAM to the P store (D receives KV
    over NIXL), so this restores store capacity without touching the engines."""
    r = Run(run, 'd')
    assert socket.gethostname() == HOSTS['d'] and not r.alive('mooncake_client') and port_free(50062)
    seg = json.loads((r.out / 'start-d.json').read_text())['mc_client_seg']
    r.spawn('mooncake_client', mc_client_cmd(seg), base_env())
    print('RESTARTED-MC-CLIENT', r.children['mooncake_client']['pid'])


def start_d(run, args):
    k = knobs(args)
    name, gpus, numa, port, rpc, master, nixl = D_ROW
    node = os.environ.get('PB_D_NODE', 'd')
    host_ip = D_HOST if node == 'd' else P_HOST
    guard(node, [port, rpc, master] + [nixl + i for i in range(8)], float(k.pop('MIN_AVAIL_GIB', 1200)))
    r = Run(run, node)
    env = base_env()
    seg = k.pop('MC_CLIENT_SEG', '1000GB')
    if seg != '0':
        r.spawn('mooncake_client', mc_client_cmd(seg), env)
    dyn = k.pop('DYN', '0') == '1'
    e = {**env, 'ROLE': 'decode', 'GPUS': gpus, 'NUMA': numa, 'PORT': str(port), 'RPC_PORT': str(rpc),
         'DP_MASTER_PORT': str(master), 'NIXL_SIDE_CHANNEL_PORT': str(nixl), 'NIXL_HOST': host_ip, 'DP': '8', **k}
    if dyn:
        e['GRPC_PORT'] = str(GRPC['decode'])
    r.spawn(name, ['bash', str(RT / 'serve.sh')], e)
    save(r.out / 'start-d.json', {'time': now(), 'knobs': k, 'mc_client_seg': seg})
    print('STARTED-D', json.dumps({n: c['pid'] for n, c in r.children.items()}))


def dyn_env(node, k):
    """Environment of the Dynamo processes: etcd discovery on the P host, TCP request plane, ZMQ events, and this
    node's routable address for everything a peer connects back to (as canlin's dyn_env_x.sh)."""
    my = P_HOST if node == 'p' else D_HOST
    ov = PB / k.get('OV', 'ov-p18' if node == 'p' else 'ov-d18')
    return {**base_env(), 'PYTHONPATH': f"{k.get('DYN_SITE', DYN_SITE)}:{ov}:{W / 'nixl-overlay'}:{PB / 'mcpath'}",
            'DYN_DISCOVERY_BACKEND': 'etcd', 'ETCD_ENDPOINTS': f'http://{P_HOST}:{ETCD_PORT}',
            'DYN_REQUEST_PLANE': 'tcp', 'DYN_EVENT_PLANE': 'zmq', 'DYN_TCP_RPC_HOST': my,
            'DYN_TCP_RESPONSE_STREAM_HOST': my, 'DYN_EVENT_PLANE_HOST': my, 'DYN_LOG': k.get('DYN_LOG', 'info'),
            'HF_HUB_OFFLINE': '1'}


def sidecar_cmd(name, mode):
    cmd = [str(V / 'bin/python'), '-m', 'dynamo.vllm.sidecar', '--grpc-endpoint', f'127.0.0.1:{GRPC[name]}',
           '--disaggregation-mode', mode, '--grpc-startup-deadline-secs', '3600',
           '--dyn-tool-call-parser', 'deepseek_v41', '--dyn-reasoning-parser', 'deepseek_v41']
    return cmd + (['--component', 'prefill'] if mode == 'prefill' else [])


def spawn_frontend(r, env, k):
    """The Dynamo HTTP frontend (KV router) on this node, port PROXY_PORT. FRONTEND_CPUS=list (taskset), FRONTEND_THREADS /
    FRONTEND_BLOCKING (Tokio worker / blocking-pool threads) bound it: by default it sizes Tokio for every CPU of the
    host (224 workers + up to 512 blocking threads) and took ~84 cores at the +300 s AgentX burst (B38)."""
    assert port_free(PROXY_PORT)
    # Frontend settings of canlin's best3 (chain_mc_node30.sh: ROUTER_AFFINITY=0, up2p1d_node30.sh env/flags).
    fe = {**env, 'DYN_ROUTER_SESSION_AFFINITY_MODE': 'soft', 'DYN_ROUTER_TEMPERATURE': '0',
          'DYN_ROUTER_QUEUE_THRESHOLD': 'None', 'DYN_TOKENIZER': 'fastokens', 'DYN_TOKENIZER_CACHE': '1',
          'DYN_TOKENIZER_CACHE_BYTES': '8000000000', 'DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS': '3600',
          'DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT': '10000000', 'DYN_TCP_CHANNEL_BUFFER': '128',
          'DYN_TCP_REQUEST_TIMEOUT': '60'}
    cmd = [str(V / 'bin/python'), '-m', 'dynamo.frontend', '--http-port', str(PROXY_PORT), '--router-mode', 'kv',
           '--active-decode-blocks-threshold', 'None', '--active-prefill-tokens-threshold', 'None',
           '--active-prefill-tokens-threshold-frac', 'None', '--no-kv-events',
           '--router-policy-config', str(RT / k.get('ROUTER_POLICY', 'router_policy.yaml'))]
    if k.get('ROUTER_AFFINITY', '0') == '1':
        cmd += ['--router-session-affinity-ttl-secs', '14400']
    # Router experiment knobs read by our Dynamo build (dynamo-site-st: shared prefix store tier for prefill selection).
    if k.get('DYN_SHARED_STORE_PENALTY_TOKENS'):
        fe['DYN_SHARED_STORE_PENALTY_TOKENS'] = k['DYN_SHARED_STORE_PENALTY_TOKENS']
    if k.get('FRONTEND_THREADS'):
        fe['DYN_RUNTIME_NUM_WORKER_THREADS'] = k['FRONTEND_THREADS']
    if k.get('FRONTEND_BLOCKING'):
        fe['DYN_RUNTIME_MAX_BLOCKING_THREADS'] = k['FRONTEND_BLOCKING']
    if k.get('FRONTEND_CPUS'):
        cmd = ['taskset', '-c', k['FRONTEND_CPUS']] + cmd
    r.spawn('frontend', cmd, fe)
    return cmd


def start_dyn(run, node, args):
    """FRONTEND_NODE=p|d (default p) places the frontend (and, via e2e_launch.sh, the AgentX client) on the P or the D
    host; InferenceX's multi-node AgentX recipes keep both off the prefill hosts. etcd and the P sidecars stay on P."""
    k = knobs(args)
    assert socket.gethostname() == HOSTS[node], socket.gethostname()
    r = Run(run, node)
    env = dyn_env(node, k)
    fe_node = k.get('FRONTEND_NODE', 'p')
    assert fe_node in ('p', 'd'), fe_node
    if node == 'd':
        r.spawn('sidecar_decode', sidecar_cmd('decode', 'decode'), env)
        if fe_node == 'd':
            cmd = spawn_frontend(r, env, k)
            save(r.out / 'start-dyn-d.json', {'time': now(), 'knobs': k, 'frontend': cmd})
        print('STARTED-DYN-D', json.dumps({n: r.children[n]['pid'] for n in r.children
                                           if n in ('frontend', 'sidecar_decode')}))
        return
    assert port_free(ETCD_PORT) and port_free(ETCD_PEER)
    data = Path(f'/tmp/etcd-pb-{run}')
    assert not data.exists(), f'{data} exists'
    r.spawn('etcd', [str(ETCD), '--name', 'pb', '--data-dir', str(data),
                     '--listen-client-urls', f'http://0.0.0.0:{ETCD_PORT}',
                     '--advertise-client-urls', f'http://{P_HOST}:{ETCD_PORT}',
                     '--listen-peer-urls', f'http://127.0.0.1:{ETCD_PEER}',
                     '--initial-advertise-peer-urls', f'http://127.0.0.1:{ETCD_PEER}',
                     '--initial-cluster', f'pb=http://127.0.0.1:{ETCD_PEER}'], base_env())
    time.sleep(5)
    assert r.alive('etcd'), 'etcd exited'
    for name, *_ in P_ROWS:
        r.spawn(f'sidecar_{name}', sidecar_cmd(name, 'prefill'), env)
    cmd = spawn_frontend(r, env, k) if fe_node == 'p' else None
    save(r.out / 'start-dyn-p.json', {'time': now(), 'knobs': k, 'frontend': cmd, 'frontend_node': fe_node})
    print('STARTED-DYN-P', json.dumps({n: r.children[n]['pid'] for n in r.children
                                       if n in ('etcd', 'frontend') or n.startswith('sidecar')}))


def status(run, node):
    r = Run(run, node)
    try:
        used, apps = gpu_rows()
    except subprocess.TimeoutExpired:
        used, apps = None, None
    out = {'host': socket.gethostname(), 'time': now(), 'gpu_mib': used,
           'alive': {n: r.alive(n) for n in r.children}}
    ip = NODE_IP[node]
    urls = {'prefill0': f'http://{ip}:{P_ROWS[0][3]}', 'prefill1': f'http://{ip}:{P_ROWS[1][3]}',
            'proxy': f'http://127.0.0.1:{PROXY_PORT}', 'frontend': f'http://127.0.0.1:{PROXY_PORT}'} \
        if node in ('p', 'p24') else {'decode': f'http://127.0.0.1:{D_ROW[3]}',
                                      'frontend': f'http://127.0.0.1:{PROXY_PORT}'}
    out['healthy'] = {n: health(u) for n, u in urls.items() if n in r.children}
    out['mem_avail_gib'] = round(mem_available_gib())
    print('STATUS', json.dumps(out))
    for n in r.children:
        log = Path(r.children[n]['log'])
        if log.exists():
            tail = log.read_bytes()[-1500:].decode(errors='replace').splitlines()[-4:]
            print(f'--- {n}:', *[t[:240] for t in tail], sep='\n  ')


def stop(run, node):
    r = Run(run, node)
    assert socket.gethostname() == HOSTS[node]
    groups = {c['pid'] for c in r.children.values()}
    logs = {c['log'] for c in r.children.values()}
    members = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            pid = int(proc.name)
            if os.getpgid(pid) not in groups or procstat(pid)[0] == 'Z':
                continue
            assert proc.stat().st_uid == os.getuid()
            # setproctitle (e.g. VLLM::DPCoordinator) clobbers /proc/<pid>/environ, so the
            # run marker or a stdout that is one of this run's logs both identify our processes.
            environ = (proc / 'environ').read_bytes().split(b'\0')
            fd1 = os.readlink(proc / 'fd/1')
            assert f'PB_RUN={run}'.encode() in environ or fd1 in logs, (pid, fd1, 'not this run')
            members.append(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    # DeepGEMM may start an MPS control daemon outside our process group; quit it only when every
    # client it serves belongs to this run.
    mps = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid() or not (proc / 'comm').read_text().startswith('nvidia-cuda-mps'):
                continue
            env = dict(x.split('=', 1) for x in (proc / 'environ').read_bytes().decode().split('\0') if '=' in x)
            pipe = env.get('CUDA_MPS_PIPE_DIRECTORY', '')
            if not pipe.startswith('/tmp/deep_gemm_mps_'):
                continue
            q = subprocess.run(['/usr/bin/nvidia-cuda-mps-control'], input=f'get_client_list {proc.name}\n',
                               capture_output=True, text=True, timeout=5, env={**os.environ, 'CUDA_MPS_PIPE_DIRECTORY': pipe})
            clients = [int(x) for x in q.stdout.split()] if q.returncode == 0 else []
            if clients and all(os.getpgid(c) in groups for c in clients):
                mps.append(pipe)
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    for g in groups:
        try:
            os.killpg(g, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(15)
    for pipe in mps:
        subprocess.run(['/usr/bin/nvidia-cuda-mps-control'], input='quit\n', capture_output=True, text=True,
                       timeout=5, env={**os.environ, 'CUDA_MPS_PIPE_DIRECTORY': pipe})
    for g in groups:
        try:
            os.killpg(g, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for _ in range(60):
        try:
            assert_free()
            break
        except AssertionError:
            time.sleep(2)
    try:
        used, apps = gpu_rows()
    except subprocess.TimeoutExpired:
        used, apps = None, None
    save(r.out / f'release-{node}-{int(time.time())}.json', {'time': now(), 'members': members, 'mps': mps,
                                                             'gpu_mib': used, 'apps': apps})
    print('STOPPED', node, 'members', len(members), 'gpu', used, 'apps', apps)


if __name__ == '__main__':
    cmd, run, *rest = sys.argv[1:]
    {'start-p': lambda: start_p(run, rest), 'start-d': lambda: start_d(run, rest),
     'start-proxy': lambda: start_proxy(run, rest), 'start-dyn': lambda: start_dyn(run, rest[0], rest[1:]),
     'restart-mc-client': lambda: restart_mc_client(run),
     'stop': lambda: stop(run, rest[0]), 'status': lambda: status(run, rest[0])}[cmd]()
