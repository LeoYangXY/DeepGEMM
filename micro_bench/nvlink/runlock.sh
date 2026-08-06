#!/usr/bin/env bash
# GPU 独占锁: 多个并行开发的 agent 都通过它跑 benchmark,
# 保证任一时刻只有一个进程在用 GPU, 否则测出来的带宽/延迟全是噪声。
# 用法: ./runlock.sh ./bin/nv01_topo [args...]
set -u
LOCK=/tmp/.nvlink_bench.lock
exec 9>"$LOCK"
if ! flock -w 900 9; then
  echo "[runlock] 等待 GPU 锁超时(900s)" >&2
  exit 1
fi
"$@"
rc=$?
flock -u 9
exit $rc
