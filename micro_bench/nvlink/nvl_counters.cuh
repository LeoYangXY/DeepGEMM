// ============================================================================
// nvl_counters.cuh —— 读 NVLink 硬件字节计数器
//
// 这是整套实验里最"硬"的一把尺子：
//   Data bytes = 协议净荷（真正的用户数据）
//   Raw  bytes = 线上总字节（净荷 + 包头 + flow-control flit + CRC + idle）
//   Raw/Data   = 协议开销比 ——> 可以反推 packet 大小 / header 开销
//
// 实现走 nvidia-smi 文本解析（driver 535 上稳定可用），
// 同时尝试 NVML field 接口（新驱动才有该 field id）。
// 计数器是全卡累积值，用法是 snapshot -> 跑负载 -> snapshot -> 相减。
// ============================================================================
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

struct NvlCounters {
  unsigned long long dataTx = 0, dataRx = 0, rawTx = 0, rawRx = 0;  // 单位 Byte
  int nlink = 0;
};

// 解析 `nvidia-smi nvlink -g{t} {d|r} -i dev`
static inline void nvl_parse(const char* cmd, unsigned long long* tx,
                             unsigned long long* rx, int* nlink,
                             const char* kTx, const char* kRx) {
  *tx = *rx = 0;
  if (nlink) *nlink = 0;
  FILE* f = popen(cmd, "r");
  if (!f) return;
  char line[512];
  while (fgets(line, sizeof line, f)) {
    char* p;
    unsigned long long v;
    if ((p = strstr(line, kTx))) {
      if (sscanf(p + strlen(kTx), " %llu", &v) == 1) { *tx += v * 1024ull; if (nlink) (*nlink)++; }
    } else if ((p = strstr(line, kRx))) {
      if (sscanf(p + strlen(kRx), " %llu", &v) == 1) *rx += v * 1024ull;
    }
  }
  pclose(f);
}

static inline NvlCounters nvl_read_counters(int dev) {
  NvlCounters c;
  char cmd[256];
  snprintf(cmd, sizeof cmd, "nvidia-smi nvlink -gt d -i %d 2>/dev/null", dev);
  nvl_parse(cmd, &c.dataTx, &c.dataRx, &c.nlink, "Data Tx:", "Data Rx:");
  snprintf(cmd, sizeof cmd, "nvidia-smi nvlink -gt r -i %d 2>/dev/null", dev);
  nvl_parse(cmd, &c.rawTx, &c.rawRx, nullptr, "Raw Tx:", "Raw Rx:");
  return c;
}

static inline NvlCounters nvl_diff(const NvlCounters& a, const NvlCounters& b) {
  NvlCounters d;
  d.dataTx = b.dataTx - a.dataTx;
  d.dataRx = b.dataRx - a.dataRx;
  d.rawTx = b.rawTx - a.rawTx;
  d.rawRx = b.rawRx - a.rawRx;
  d.nlink = b.nlink;
  return d;
}

// 打印一次测量的 wire 分析
//   payload  = 你的 kernel 逻辑上搬了多少字节
static inline void nvl_report(const char* tag, double payload,
                              const NvlCounters& d) {
  printf("%-18s payload=%8.1f MB | dataTx=%8.1f dataRx=%8.1f rawTx=%8.1f "
         "rawRx=%8.1f MB | raw/data=%5.3f  wire/payload=%5.3f\n",
         tag, payload / 1e6, d.dataTx / 1e6, d.dataRx / 1e6, d.rawTx / 1e6,
         d.rawRx / 1e6,
         (d.dataTx + d.dataRx) ? (double)(d.rawTx + d.rawRx) / (d.dataTx + d.dataRx) : 0.0,
         payload ? (double)(d.rawTx + d.rawRx) / payload : 0.0);
}
