# Hardware extension analysis preparation

状态：2026-09-08 新 RTX 4090 的 microbench、profile 与 Triton 真实产物均已通过各自离线审计；微基准另完成 SM89 指令人工审查和独立硬件计数器对照，见 [证据索引](../evidence-index.md)。持续峰值和硬件 roofline 仍未确立。这些新文件不属于冻结上传包，不应补传或用于重建已冻结测量源码。

## 已读取的产物与证据边界

- `cuda_microbench`：首行 environment，随后 2 个操作 × 2 个 block × 2 个 scale × 5 批 × 30 样本，共 1200 行。environment 只有 GPU 名称、L2、SM 数量；UUID、源码和可执行文件 SHA 由下方外层采集器记录。每个配置仅在预热后核对前 16 个输出，不能称全输出正确性验证。
- `profile_session`：`receipt.json` 保存一次 probe 加 30 个正式 capture、2 个 unsupported 组合；每项绑定 CSV、ncu-rep、target receipt SHA、GPU、源码树、库和 correctness SHA。target 含前后全 FP32/64 FP64 点检、PID 和一次逻辑调用；CSV PID 必须与 target 一致。只完成 probe 不等于完成 30 个组合。
- Triton：check/tune/evaluate 各自 manifest，shapes 分片及 PTX；tune 另输出 frozen.json。evaluate 的 CUDA 和 Triton 分别跑完整 suite，不能将双方 batch 编号当作配对采样。5 批仅供描述统计，不按 PLAN 的十批门槛发布确定 WIN/LOSS。

## 可直接使用的外层采集

先在已授权 GPU 主机设置 `FROZEN_ROOT` 为实际解包的冻结源码目录、`MICRO_BINARY` 为该目录构建的 `cuda_microbench`、`MICRO_OUT` 为源码目录之外的全新结果目录。下述代码只执行已构建的微基准、查询工具和设备并写新结果；不修改源码、权限、驱动或时钟，不安装软件。默认完整 microbench 上限 180 秒，各工具调用上限 60 秒。微基准失败会停止；工具超时保留 TIMEOUT，工具缺失或非零退出会记录而不伪称指令审查通过，已有日志保留；两个 nvidia-smi 观测必须只有同一张 GPU。它不声称证明独占性或编译过程，应一并保存原 CMake 配置/编译日志。

```sh
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import datetime, hashlib, json, os, pathlib, shutil, subprocess, sys
root = pathlib.Path(os.environ['FROZEN_ROOT']).resolve()
binary = pathlib.Path(os.environ['MICRO_BINARY']).resolve()
out = pathlib.Path(os.environ['MICRO_OUT']).resolve()
assert not out.is_relative_to(root) and binary.is_file()
out.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(root))
from bench.run import source_identity

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def gpu():
    value = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                                    '--format=csv,noheader'], text=True, timeout=10)
    rows = [r.strip() for r in value.splitlines() if r.strip()]
    assert len(rows) == 1, 'requires one physical GPU'
    name, uuid, driver = [v.strip() for v in rows[0].split(',')]
    return dict(name=name, uuid=uuid, driver_version=driver)

m = dict(kind='microbench_capture_v1', smoke=False, returncode=None,
         started_utc=now(), source=source_identity(), binary_sha256=sha(binary),
         binary_path=str(binary), gpu_before=gpu(), gpu_after=None, artifacts={}, commands=[])
def save(): (out/'manifest.json').write_text(json.dumps(m, indent=2)+'\n')
def run(name, args, seconds=60):
    row=dict(name=name, command=args, started_utc=now(), returncode=None,
             stdout=name+'.stdout.txt', stderr=name+'.stderr.txt')
    path, err = out/row['stdout'], out/row['stderr']
    try:
        with path.open('w') as stdout, err.open('w') as stderr:
            if not shutil.which(args[0]):
                row['status']='TOOL_UNAVAILABLE'
            else:
                try:
                    p=subprocess.run(args,stdout=stdout,stderr=stderr,timeout=seconds)
                    row.update(returncode=p.returncode,status='PASS' if p.returncode==0 else 'FAIL')
                except subprocess.TimeoutExpired:
                    row['status']='TIMEOUT'
    finally:
        row.update(finished_utc=now(),stdout_sha256=sha(path),stderr_sha256=sha(err))
        m['commands'].append(row); save()
    return row
save()
try:
    assert len(m['source']['source_files'])==71
    assert m['source']['source_tree_sha256']=='5b499cd77b1594e501fce8c8d457ec147ffe410ae5764fa6b1df1a4566934cbd'
    for name, args in [('nvcc-version',['nvcc','--version']),
                       ('cuobjdump-version',['cuobjdump','--version']),
                       ('nvdisasm-version',['nvdisasm','--version']),
                       ('ncu-version',['ncu','--version']),
                       ('ptx',['cuobjdump','--dump-ptx',str(binary)]),
                       ('sass',['cuobjdump','--dump-sass',str(binary)]),
                       ('resources',['cuobjdump','--dump-resource-usage',str(binary)])]:
        row=run(name,args)
        if row['status']=='PASS' and (out/row['stdout']).stat().st_size and name in ('ptx','sass','resources'):
            m['artifacts'][name]=dict(path=row['stdout'],sha256=row['stdout_sha256'])
    def observe(phase):
        run('telemetry-'+phase,['nvidia-smi','--query-gpu=uuid,temperature.gpu,power.limit,power.draw,clocks.sm','--format=csv'])
        run('processes-'+phase,['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'])
    observe('before')
    row=run('microbench',[str(binary)],180)
    m.update(returncode=row['returncode'],raw_sha256=row['stdout_sha256'],
             microbench_started_utc=row['started_utc'],microbench_finished_utc=row['finished_utc'])
    observe('after')
    m.update(gpu_after=gpu(),binary_sha256_after=sha(binary),source_after=source_identity())
    assert m['gpu_before']==m['gpu_after'], 'GPU identity changed'
    assert m['binary_sha256']==m['binary_sha256_after'], 'binary changed'
    assert m['source']==m['source_after'], 'source changed'
    assert row['status']=='PASS', 'microbench failed'
except BaseException as e:
    m['error']=repr(e)
    raise
finally:
    m['finished_utc']=now(); save()
PY
```

工具缺失保留 `TOOL_UNAVAILABLE`，不等于 SASS 通过。可由根代理按既有授权核对官方 NVIDIA CUDA 12.8 apt 仓库的 `cuda-cuobjdump-12-8`、`cuda-nvdisasm-12-8` 候选及依赖后安装最小工具；此文档不执行安装，不使用会升级驱动的 `cuda` 大包。以主机实际 `--help` 为准；[官方 Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)确认 cuobjdump 可直接读取宿主可执行文件，nvdisasm 需要 cubin。

## PTX/SASS 人工核验

采集器已执行下列对**实际测量二进制**的命令。还应对测量使用的 `libgroupconv` 同样执行，并单独保存二进制 SHA、命令、工具版本与输出 SHA：

```sh
cuobjdump --dump-ptx "$MICRO_BINARY" > microbench.ptx.txt
cuobjdump --dump-sass "$MICRO_BINARY" > microbench.sass.txt
cuobjdump --dump-resource-usage "$MICRO_BINARY" > microbench.resources.txt
rg -n 'eight_fma_chains|logical_copy|FFMA|fma.rn.f32|LDG|STG|BRA' microbench.ptx.txt microbench.sass.txt
```

`rg` 只是定位，不是验收器。对实际 GPU 所选架构的 `eight_fma_chains` 检查循环回边、迭代次数、8 个独立累加寄存器及最终写出；若循环展开，应按展开倍数还原动态 FMA 数，不能简单要求静态恰好 8 条 FFMA。确认无被消除的链、意外降精度或大规模 spill，并记录寄存器使用。计量为 `threads × iterations × 8 × 2`，不包含初始化/最终归约指令。PTX 存在不证明其是最终执行 SASS；若 binary 不嵌入 PTX，记录缺失，不用重新编译的产物冒充测量二进制。

copy 核查全局读写和 grid-stride 循环；每个输入 buffer 至少 4×L2，逻辑量按读+写 `2×buffer_bytes`。该速度仍是 logical copy bandwidth，不自动等于 counter-confirmed DRAM bandwidth。保留全部 8 配置和批中位数；FMA 的 source-formula rate 在指令审查前不得当 roofline 算力上界。五批和未锁频的状态须保留。

## 离线运行及 manifest 接口

```sh
python scripts/analyze_hardware_extensions.py \
  --raw DOWNLOADED/microbench.stdout.txt \
  --manifest DOWNLOADED/manifest.json \
  --snapshot EXTRACTED_FROZEN_SOURCE \
  --binary DOWNLOADED/cuda_microbench \
  --output NEW_LOCAL_ANALYSIS/microbench-analysis.json
```

必填 manifest 为 `kind=microbench_capture_v1`、`returncode=0`、`smoke=false`、raw/binary SHA、闭合的 commands（argv/状态/严格整数退出码/带时区起止/stdout和stderr文件及SHA）、整轮与microbench起止、前后telemetry/processes查询记录、完全相等且非空 UUID 的 gpu_before/gpu_after、source_identity 格式的 source，以及与运行前一致的 source_after/binary_sha256_after。可选 artifacts 的 ptx/sass/resources 为 manifest 目录内相对路径及 SHA。分析器只接受已审查 microbench.cu SHA `8242bbe4412e26d94a407030fca6c5a90d47ad6cd3b282837031a3adc9b27b10`，逐项重算字节/FMA 公式并验证完整 8×150 分母、scope、正有限时间及唯一索引。产物始终为描述统计，不自动发布可持续峰值、DRAM roofline 或人工 SASS 通过状态。合成测试可通过收据一致性审计，因此 `audit_status=PASS` 绝不是执行真实性证明。

## Profile / roofline 核验范围

1. 先校验下载 manifest，重算 receipt、target、csv、ncu-rep SHA；对冻结 source_files 逐文件验算，并核对 library/native correctness/protocol SHA。require phase=collect、status=PASS、counter_gate=PASS、30 个唯一正式组合加 probe、2 个明确 unsupported；不得拿 probe 替正式项。
2. 所有 capture 与 target 的 GPU UUID/设备身份相同；与 microbench 外层 UUID、driver、source tree 相同才可讨论同卡模型。保留软件版本、nvidia-smi 频温功耗、时间戳和独占性未验证状态。同 UUID 也不能抹去不同时段频率/功率条件差异。
3. 重新导出 `.ncu-rep`：`ncu --config-file off --import REPORT.ncu-rep --csv --page raw --print-units base`。按 [官方 CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)及本机 help 核验选项；复用 frozen `parse_counter_csv` 检查单 kernel/PID、核名、必需 metric、非 N/A、单位和重复项。检查 profile 前后数值证据，不能只看 status=true。
4. 使用**同一个 capture** 的 `dram__bytes_read.sum + dram__bytes_write.sum` 得到 B_dram_profile；将 `gpu__time_duration.sum` 按实际单位换秒后计算 DRAM B/s。`I_dram=F_nominal/B_dram_profile`、`P_nominal=F_nominal/t_profile`；F_nominal 来自冻结 shape。零 traffic 不产生无穷强度，标无法计算。缓存清空的 profile 时间不得与普通 graph/eager 时间拼接。
5. 首轮只分析 direct k0–k3。F_nominal 包含 padding 名义工作，不代表动态执行 FMA；不命名为真实算力利用率。L2 sector 不可直接当字节，需确认架构 sector 单位，且 L2 ceiling 必须独立测量。当前 copy/FMA 是稳态微基准，与 cache-all profile 条件不同；即使同卡同源，也须披露差异，不能画一条线便宣称瓶颈已证实。

## Triton 冻结 / holdout 验收范围

核对 check/tune/evaluate 的 binding 五项（source tree、protocol、split、native library、native correctness）和完整 environment；按下载映射重定位远程绝对路径并校验原 SHA，禁止直接读取不存在的远程路径或忽略校验。保留原 receipt，不能为方便改写它后沿用旧 hash。

check 必须包含所有 boundary shapes、全部候选 blocks、全部正确性 seeds；supported 为完整 FP32 与 full-output FP64，unsupported 仍占分母。tune 只读冻结 tuning ID 集，检查每个候选 5×30、PTX SHA 和编译 metadata；用原始 graph 批中位数重新计算按 cpg=1/2/4 的几何平均选块及平分时最小 block 规则。frozen 的 evidence 必须闭合 jobs/check/PTX，frozen SHA 与 tune receipt 一致，`holdout_performance_accessed=false`；这个声明还须执行顺序回执佐证。

evaluate 仅使用对应 cpg 的冻结 block，核对全部 holdout ID 和 unsupported 分母；不得按 holdout 最优 block 回改 frozen。CUDA 与 Triton suite 不是配对批次，不做跨 suite paired bootstrap；5 批比值明确描述性、非确定获胜。保留 first-call JIT+launch（含可能的轮内 cache reuse），不能称纯编译耗时。PTX 需人工检查 explicit FP32 FMA、掩码和目标架构，未保存执行 cubin/SASS 则其状态仍待核验。

修订门禁：生产审计固定 031 的 71 文件及树哈希 `5b499cd77b1594e501fce8c8d457ec147ffe410ae5764fa6b1df1a4566934cbd`，不接受单文件自行构造的 source_identity。所有命令日志必须保留；汇编产物必须与成功命令的同 binary argv、stdout SHA 一致。频率/温度/功率限制/使用及进程观测保存原始 CSV 与失败状态，数值 N/A 不自动转换为零；缺失能力不代表独占通过。
